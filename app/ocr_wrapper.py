from __future__ import annotations

from .image_ops import ensure_color
from .models import OcrAttempt, OcrLine
from .utils import dedup_lines, safe_float

try:
    from paddleocr import PaddleOCR
except Exception as e:  # pragma: no cover
    PaddleOCR = None
    _PADDLE_IMPORT_ERROR = e
else:
    _PADDLE_IMPORT_ERROR = None


class OCRWrapper:
    def __init__(self, lang: str = 'ch', use_angle_cls: bool = True, use_gpu: bool = False):
        if PaddleOCR is None:
            raise ImportError(f'导入 PaddleOCR 失败：{_PADDLE_IMPORT_ERROR}')
        self.engine = PaddleOCR(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu, show_log=False)

    def ocr_image(self, image, tag: str) -> OcrAttempt:
        image = ensure_color(image)
        result = self.engine.ocr(image, cls=True)
        lines = []
        if isinstance(result, list):
            for item in result:
                if not item:
                    continue
                if isinstance(item, list):
                    for det in item:
                        try:
                            text = det[1][0]
                            score = safe_float(det[1][1], 0.0)
                        except Exception:
                            continue
                        lines.append(OcrLine(text=str(text), score=score))
        return OcrAttempt(tag=tag, lines=dedup_lines(lines))
