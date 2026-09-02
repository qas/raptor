# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


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
    hiddenimports=[
        "raptor.chat.providers.telegram",
        "raptor.chat.providers.responses_provider",
        "raptor.chat.providers.multi_provider",
        "raptor.shell.shell_supervisor",
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
