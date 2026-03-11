from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .constants import BAD_NAME_WORDS
from .models import OcrLine


def normalize_text(text: str) -> str:
    if not text:
        return ''
    text = text.replace('\u3000', ' ').replace('：', ':').replace('。', '.').replace('，', ',')
    text = re.sub(r'\s+', '', text)
    return text.strip()


def sanitize_name_for_fs(name: str) -> str:
    name = (name or '').strip()
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    name = re.sub(r'\s+', '', name)
    return name or '未命名'


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 1
    while True:
        cand = path.parent / f'{path.name}__{idx}'
        if not cand.exists():
            return cand
        idx += 1


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def dedup_lines(lines: Iterable[OcrLine]) -> List[OcrLine]:
    best: Dict[str, OcrLine] = {}
    for line in lines:
        key = normalize_text(line.text)
        if not key:
            continue
        old = best.get(key)
        if old is None or line.score > old.score:
            best[key] = OcrLine(text=line.text.strip(), score=float(line.score))
    return list(best.values())


def count_hits(text: str, keywords: Sequence[str]) -> int:
    t = normalize_text(text)
    return sum(1 for k in keywords if k in t)


def parse_std_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip().replace('年', '.').replace('月', '.').replace('日', '')
    s = s.replace('/', '.').replace('-', '.')
    s = re.sub(r'[^0-9.]', '', s)
    s = re.sub(r'\.+', '.', s).strip('.')
    m = re.fullmatch(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', s)
    if not m:
        m = re.fullmatch(r'(\d{4})(\d{2})(\d{2})', s)
    if not m:
        return None
    y, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y < 1949 or y > 2100:
        return None
    try:
        dt = date(y, mm, dd)
    except Exception:
        return None
    return dt.strftime('%Y.%m.%d')


def extract_id_number(text: str) -> Optional[str]:
    t = normalize_text(text)
    m = re.search(r'(?<!\d)(\d{17}[0-9Xx])(?!\d)', t)
    return m.group(1).upper() if m else None


def year_of(d: str) -> int:
    return int(d.split('.')[0])


def is_plausible_validity(start: str, end: str) -> bool:
    if not start or not end:
        return False
    if end == '长期':
        try:
            sy = year_of(start)
            return 1949 <= sy <= 2100
        except Exception:
            return False
    try:
        ds = datetime.strptime(start, '%Y.%m.%d').date()
        de = datetime.strptime(end, '%Y.%m.%d').date()
    except Exception:
        return False
    delta = (de - ds).days
    return 0 <= delta <= 365 * 50


def validity_bucket(valid_to: Optional[str]) -> str:
    if not valid_to or valid_to == '未识别结束':
        return '未识别有效期'
    if valid_to == '长期':
        return '未过期'
    try:
        dt = datetime.strptime(valid_to, '%Y.%m.%d').date()
    except Exception:
        return '未识别有效期'
    return '已过期' if dt < date.today() else '未过期'


def normalize_name_candidate(s: str) -> str:
    s = s.strip()
    s = re.sub(r'^[姓名:：\s]+', '', s)
    s = re.sub(r'[^\u4e00-\u9fff·]', '', s)
    return s.strip('·')


def is_bad_name(s: str) -> bool:
    if not s:
        return True
    if s in BAD_NAME_WORDS or any(w in s for w in BAD_NAME_WORDS):
        return True
    return bool(re.search(r'[年月日省市县区镇乡村路街号男女性别民族出生住址公民身份证机关期限银行公司寿联]', s))
