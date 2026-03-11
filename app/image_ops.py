from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.core.multiarray
import cv2


def read_image_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f'无法读取图片字节: {path}')
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'cv2.imdecode 失败: {path}')
    return img


def ensure_color(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def rotate_bound(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    m[0, 2] += (new_w / 2) - center[0]
    m[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def clahe_enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([((i / 255.0) ** inv) * 255 for i in np.arange(256)]).astype('uint8')
    return cv2.LUT(image, table)


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def binarize(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


def upscale_for_ocr(image: np.ndarray, target_short: int = 1200) -> np.ndarray:
    h, w = image.shape[:2]
    short_side = min(h, w)
    if short_side <= 0 or short_side >= target_short:
        return image
    scale = target_short / short_side
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def preprocess_variants(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    return [
        ('orig', image),
        ('rot180', rotate_bound(image, 180)),
        ('clahe', clahe_enhance(image)),
        ('sharp', sharpen(image)),
        ('gamma1.25', adjust_gamma(image, 1.25)),
        ('bin', binarize(image)),
        ('up', upscale_for_ocr(image)),
    ]
