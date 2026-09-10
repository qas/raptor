# -*- mode: python ; coding: utf-8 -*-
import os
import tempfile
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


# Analysis imports application, which loads state at import time.
os.environ["RAPTOR_HOME"] = tempfile.mkdtemp(prefix="raptor-pyi-home-")
os.environ["AGENT_WORKDIR"] = tempfile.mkdtemp(prefix="raptor-pyi-work-")


def _package_modules(package_dir: Path) -> list[str]:
    root = package_dir.parent
    names: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(root)
        parts = (
            relative.parent.parts
            if path.name == "__init__.py"
            else relative.with_suffix("").parts
        )
        if parts:
            names.append(".".join(parts))
    return names


build_data = [
    ("pyproject.toml", "."),
    ("assets/skills", "assets/skills"),
]
if Path(".raptor-build.json").is_file():
    build_data.append((".raptor-build.json", "."))

a = Analysis(
    ["raptor.py"],
    pathex=[],
    binaries=[],
    datas=collect_data_files("certifi") + build_data,
    hiddenimports=_package_modules(Path(SPECPATH) / "raptor") + [
        "socksio",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="raptor",
    debug=False,
    bootloader_ignore_signals=True,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="raptor",
)
