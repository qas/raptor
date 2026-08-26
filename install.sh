#!/bin/sh
set -eu

REPO="${RAPTOR_REPO:-qas/raptor}"
INSTALL_ROOT="${RAPTOR_INSTALL_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/raptor}"
BIN_DIR="${RAPTOR_BIN_DIR:-$HOME/.local/bin}"
VERSION="${RAPTOR_VERSION:-}"

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

is_release_version() {
    case "$1" in
        v[0-9]*.[0-9]*.[0-9]*)
            case "$1" in
                *[!v0-9.]*|*".."*|v.*|*. ) return 1 ;;
            esac
            return 0
            ;;
    esac
    return 1
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
