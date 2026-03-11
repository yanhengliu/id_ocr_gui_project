# 身份证 OCR GUI 项目说明

## 目录结构

```text
id_ocr_gui_project/
├── main.py
├── requirements.txt
├── README_BUILD.md
├── .github/
│   └── workflows/
│       └── main.yml
└── app/
    ├── __init__.py
    ├── analyzer.py
    ├── constants.py
    ├── extractors.py
    ├── gui.py
    ├── image_ops.py
    ├── logging_utils.py
    ├── models.py
    ├── ocr_wrapper.py
    ├── processor.py
    └── utils.py
```

## 本地运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Windows 打包

建议在 Windows 或 GitHub Actions 的 Windows Runner 上打包：

```bash
pip install -r requirements.txt
pyinstaller --noconfirm --clean --windowed --name IDCardOCRTool --collect-all paddleocr --collect-all pyclipper --collect-all shapely --collect-all skimage --hidden-import cv2 main.py
```

打包完成后，exe 位于：

```text
dist/IDCardOCRTool.exe
```

## 输出内容

- `身份证有效信息分类/`
  - `已过期/`
  - `未过期/`
  - `未识别有效期/`
- `logs/run_*.log`
- `logs/summary.csv`
- `logs/summary.json`
- `debug/<子文件夹>/xxx.json`
- 每个结果目录中的 `result.json`
