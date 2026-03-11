# id_ocr.spec
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    collect_dynamic_libs,
    copy_metadata,
)

datas = []
binaries = []
hiddenimports = []
runtime_hooks = ["hooks/runtime_numpy_first.py"]

# numpy
datas += collect_data_files("numpy")
binaries += collect_dynamic_libs("numpy")
hiddenimports += collect_submodules("numpy")
datas += copy_metadata("numpy")

# cv2
datas += collect_data_files("cv2")
hiddenimports += collect_submodules("cv2")

# paddle / paddleocr
datas += collect_data_files("paddle")
datas += collect_data_files("paddleocr")
binaries += collect_dynamic_libs("paddle")
hiddenimports += collect_submodules("paddle")
hiddenimports += collect_submodules("paddleocr")
datas += copy_metadata("paddlepaddle")
datas += copy_metadata("paddleocr")

# Pillow
datas += collect_data_files("PIL")
hiddenimports += collect_submodules("PIL")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IDCardOCRTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="IDCardOCRTool",
)
