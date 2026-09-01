#!/bin/sh
set -eu

REPO="${RAPTOR_REPO:-qas/raptor}"
INSTALL_ROOT="${RAPTOR_INSTALL_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/raptor}"
BIN_DIR="${RAPTOR_BIN_DIR:-$HOME/.local/bin}"
VERSION="${RAPTOR_VERSION:-}"
SANDBOX_PROBE_TIMEOUT_SECONDS=5
SANDBOX_ERROR_LIMIT_BYTES=4096

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

is_release_version() {
    [ "$1" = "nightly" ] && return 0
    version=${1#v}
    [ "$version" != "$1" ] || return 1
    core=${version%%-*}
    prerelease=
    if [ "$core" != "$version" ]; then
        prerelease=${version#*-}
        case "$prerelease" in
            alpha.*|beta.*|rc.*) ;;
            *) return 1 ;;
        esac
        stage=${prerelease%%.*}
        number=${prerelease#*.}
        [ "$stage.$number" = "$prerelease" ] || return 1
        case "$number" in
            ""|*[!0-9]*|0[0-9]*) return 1 ;;
        esac
    fi
    old_ifs=$IFS
    IFS=.
    set -- $core
    IFS=$old_ifs
    [ "$#" -eq 3 ] || return 1
    for number in "$@"; do
        case "$number" in
            ""|*[!0-9]*|0[0-9]*) return 1 ;;
        esac
    done
    return 0
}

absolute_dir() {
    mkdir -p "$1"
    (cd "$1" && pwd)
}

resolve_dir() {
    if [ -d "$1" ]; then
        (cd "$1" && pwd)
        return
    fi
    parent=$(dirname -- "$1")
    base=$(basename -- "$1")
    if [ -d "$parent" ]; then
        printf '%s/%s\n' "$(cd "$parent" && pwd)" "$base"
        return
    fi
    printf '%s\n' "$1"
}

physical_dir() {
    [ -d "$1" ] || return 1
    (cd "$1" && pwd -P)
}

is_under() {
    case "$1" in
        "$2"|"$2"/*) return 0 ;;
    esac
    return 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Raptor install requires $1"
}

linux_sandbox_install_command() {
    case "$1" in
        ubuntu|debian) printf '%s\n' "sudo apt-get install bubblewrap" ;;
        fedora|rhel|centos) printf '%s\n' "sudo dnf install bubblewrap" ;;
        arch) printf '%s\n' "sudo pacman -S bubblewrap" ;;
        *) printf '%s\n' "Install bubblewrap with your system package manager" ;;
    esac
}

linux_distribution_id() {
    os_release=${1:-/etc/os-release}
    [ -r "$os_release" ] || return 1
    awk -F= '$1 == "ID" {
        value = $2
        gsub(/^"|"$/, "", value)
        print value
        exit
    }' "$os_release"
}

ubuntu_userns_restriction_enabled() {
    os_release=${1:-/etc/os-release}
    restriction=${2:-/proc/sys/kernel/apparmor_restrict_unprivileged_userns}
    [ "$(linux_distribution_id "$os_release")" = "ubuntu" ] || return 1
    [ -r "$restriction" ] || return 1
    [ "$(tr -d '[:space:]' < "$restriction")" = "1" ]
}

trusted_linux_bwrap() {
    metadata=$(stat -Lc '%u %a' -- "$1" 2>/dev/null) || return 1
    set -- $metadata
    [ "$#" -eq 2 ] || return 1
    owner=$1
    mode=$2
    case "$owner" in
        ""|*[!0-9]*) return 1 ;;
    esac
    case "$mode" in
        ""|*[!0-7]*) return 1 ;;
    esac
    [ "$owner" -eq 0 ] || return 1
    [ $((mode / 10 % 10 & 2)) -eq 0 ] || return 1
    [ $((mode % 10 & 2)) -eq 0 ]
}

report_linux_sandbox() {
    temporary_directory=$1
    os_release=${2:-/etc/os-release}
    restriction=${3:-/proc/sys/kernel/apparmor_restrict_unprivileged_userns}
    probe_timeout=${SANDBOX_PROBE_TIMEOUT_SECONDS:-5}
    error_limit=${SANDBOX_ERROR_LIMIT_BYTES:-4096}
    if ! bwrap_path=$(command -v bwrap 2>/dev/null); then
        distribution=$(linux_distribution_id "$os_release" || true)
        printf '%s\n' \
            "Linux shell sandbox: unavailable (Bubblewrap is not installed)" \
            "Run: $(linux_sandbox_install_command "$distribution")" \
            "Raptor will fail closed if permissions.filesystem.deny_read is configured."
        return
    fi
    for utility in stat timeout head grep tr; do
        if ! command -v "$utility" >/dev/null 2>&1; then
            printf '%s\n' \
                "Linux shell sandbox: not verified (${utility} is unavailable)" \
                "Raptor will verify Bubblewrap before each restricted shell command."
            return
        fi
    done
    if ! trusted_linux_bwrap "$bwrap_path"; then
        printf '%s\n' \
            "Linux shell sandbox: unavailable (Bubblewrap is not trusted)" \
            "Bubblewrap must be root-owned and not group/world writable." \
            "Raptor will fail closed if permissions.filesystem.deny_read is configured."
        return
    fi
    sandbox_error="${temporary_directory}/bubblewrap-probe-error"
    if timeout "$probe_timeout" "$bwrap_path" \
        --ro-bind / / /bin/true 2>"$sandbox_error"; then
        printf '%s\n' "Linux shell sandbox: ready"
        return
    fi
    error=$(head -c "$error_limit" "$sandbox_error")
    printf '%s\n' "Linux shell sandbox: unavailable"
    if ubuntu_userns_restriction_enabled "$os_release" "$restriction" && \
        printf '%s' "$error" | grep -q \
            'setting up uid map: Permission denied'; then
        printf '%s\n' \
            "Ubuntu AppArmor denied Bubblewrap access to user namespaces." \
            "Setup: https://github.com/${REPO}#ubuntu-apparmor" \
            "Raptor did not weaken the host policy automatically."
    else
        printf '%s\n' "Bubblewrap probe failed: ${error:-no error output}"
    fi
    printf '%s\n' \
        "Raptor will fail closed if permissions.filesystem.deny_read is configured."
}

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1 }'
        return
    fi
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{ print $1 }'
        return
    fi
    die "Raptor install requires sha256sum or shasum"
}

installation_pid() {
    root="$1"
    [ -d "$root" ] || return 1
    if [ -d /proc ]; then
        physical=$(physical_dir "$root") || physical=$root
        for exe in /proc/[0-9]*/exe; do
            target=$(readlink "$exe" 2>/dev/null) || continue
            if is_under "$target" "$physical" || is_under "$target" "$root"; then
                pid=${exe#/proc/}
                pid=${pid%/exe}
                printf '%s\n' "$pid"
                return 0
            fi
        done
        return 1
    fi
    if command -v lsof >/dev/null 2>&1; then
        for bin in "$root"/versions/*/raptor; do
            [ -e "$bin" ] || continue
            pid=$(lsof -t "$bin" 2>/dev/null | awk 'NR==1 { print; exit }') || true
            if [ -n "${pid:-}" ]; then
                printf '%s\n' "$pid"
                return 0
            fi
        done
        return 1
    fi
    die "Cannot determine whether Raptor is running; install lsof or stop it first"
}

owned_link_target() {
    link="$1"
    prefix="$2"
    [ -L "$link" ] || return 1
    target=$(readlink "$link")
    case "$target" in
        /*) ;;
        *) target="$(dirname -- "$link")/$target" ;;
    esac
    is_under "$target" "$prefix"
}

uninstall() {
    INSTALL_ROOT=$(resolve_dir "$INSTALL_ROOT")
    BIN_DIR=$(resolve_dir "$BIN_DIR")
    versions="${INSTALL_ROOT}/versions"
    link="${BIN_DIR}/raptor"
    if pid=$(installation_pid "$INSTALL_ROOT"); then
        die "Raptor is running (pid ${pid}); stop it before uninstalling"
    fi
    if owned_link_target "$link" "$INSTALL_ROOT"; then
        rm -f "$link"
    elif [ -e "$link" ] || [ -L "$link" ]; then
        printf 'Leaving %s; it does not point at this Raptor install\n' "$link"
    fi
    if [ -e "$versions" ]; then
        rm -rf "$versions"
    fi
    if [ -d "$INSTALL_ROOT" ]; then
        rmdir "$INSTALL_ROOT" 2>/dev/null || true
    fi
    printf 'Uninstalled Raptor from %s\n' "$INSTALL_ROOT"
}

install_release() {
    os_name=$(uname -s)
    arch_name=$(uname -m)
    case "$os_name" in
        Linux) platform_os=linux ;;
        Darwin) platform_os=macos ;;
        *) die "Raptor releases support Linux and macOS only" ;;
    esac
    case "$arch_name" in
        x86_64|amd64) platform_arch=x86_64 ;;
        aarch64|arm64)
            if [ "$platform_os" = linux ]; then
                platform_arch=aarch64
            else
                platform_arch=arm64
            fi
            ;;
        *) die "Unsupported architecture: $arch_name" ;;
    esac
    require_cmd curl
    require_cmd tar
    require_cmd mktemp
    require_cmd awk
    if ! command -v sha256sum >/dev/null 2>&1 && \
        ! command -v shasum >/dev/null 2>&1; then
        die "Raptor install requires sha256sum or shasum"
    fi
    if [ -z "$VERSION" ]; then
        latest_url=$(curl -fsSL -o /dev/null -w '%{url_effective}' \
            "https://github.com/${REPO}/releases/latest")
        VERSION=${latest_url##*/}
    fi
    is_release_version "$VERSION" || \
        die "Could not resolve a Raptor release version"
    archive="raptor-${VERSION}-${platform_os}-${platform_arch}.tar.gz"
    base="https://github.com/${REPO}/releases/download/${VERSION}"
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    curl -fsSL -o "${tmp}/${archive}" "${base}/${archive}"
    curl -fsSL -o "${tmp}/SHA256SUMS" "${base}/SHA256SUMS"
    expected=$(awk -v name="$archive" '$2 == name { print $1 }' \
        "${tmp}/SHA256SUMS")
    [ -n "$expected" ] || die "No checksum for ${archive}"
    actual=$(sha256_file "${tmp}/${archive}")
    [ "$expected" = "$actual" ] || die "Checksum mismatch for ${archive}"
    tar -xzf "${tmp}/${archive}" -C "$tmp"
    [ -x "${tmp}/raptor/raptor" ] || die "Release archive is missing raptor"
    INSTALL_ROOT=$(absolute_dir "$INSTALL_ROOT")
    BIN_DIR=$(absolute_dir "$BIN_DIR")
    dest="${INSTALL_ROOT}/versions/${VERSION}"
    mkdir -p "${INSTALL_ROOT}/versions"
    rm -rf "$dest"
    mv "${tmp}/raptor" "$dest"
    link_tmp=$(mktemp "${BIN_DIR}/raptor.XXXXXX")
    rm -f "$link_tmp"
    ln -s "$dest/raptor" "$link_tmp"
    mv -f "$link_tmp" "${BIN_DIR}/raptor"
    printf 'Installed Raptor %s to %s\n' "$VERSION" "$dest"
    printf 'Command: %s\n' "${BIN_DIR}/raptor"
    if [ "$platform_os" = linux ]; then
        report_linux_sandbox "$tmp"
    fi
    case ":$PATH:" in
        *":${BIN_DIR}:"*) ;;
        *)
            printf 'Add %s to PATH to run raptor\n' "$BIN_DIR"
            ;;
    esac
}

case "${1:-}" in
    --uninstall)
        [ "$#" -eq 1 ] || die "Unexpected arguments after --uninstall"
        uninstall
        ;;
    "")
        install_release
        ;;
    *)
        die "Unknown argument: $1"
        ;;
esac
