from __future__ import annotations

import re
from typing import Dict, Optional, Sequence, Tuple

from .constants import BACK_KEYWORDS
from .utils import is_bad_name, is_plausible_validity, normalize_name_candidate, normalize_text, parse_std_date


def extract_name_strong(lines: Sequence[str], text: str) -> Optional[str]:
    candidates: Dict[str, float] = {}

    def add(name: str, score: float) -> None:
        name = normalize_name_candidate(name)
        if not name or len(name) < 2 or len(name) > 4 or is_bad_name(name):
            return
        candidates[name] = max(candidates.get(name, 0.0), score)

    raw_lines = [x.strip() for x in lines if x and x.strip()]
    norm_lines = [normalize_text(x) for x in raw_lines]

    for line in raw_lines:
        n = normalize_text(line)
        m = re.search(r'姓名[:：]?([\u4e00-\u9fff·]{2,4})', n)
        if m:
            add(m.group(1), 120.0)

    for line in raw_lines:
        cand = normalize_name_candidate(line)
        if re.fullmatch(r'[\u4e00-\u9fff·]{2,4}', cand) and not is_bad_name(cand):
            add(cand, 70.0)

    for i, line in enumerate(norm_lines[:-1]):
        if line in {'姓名', '姓名:', '姓名：'}:
            cand = normalize_name_candidate(raw_lines[i + 1])
            if re.fullmatch(r'[\u4e00-\u9fff·]{2,4}', cand) and not is_bad_name(cand):
                add(cand, 100.0)

    joined = normalize_text(text)
    for name in list(candidates.keys()):
        candidates[name] += min(joined.count(name), 3) * 10.0

    return max(candidates.items(), key=lambda kv: kv[1])[0] if candidates else None


def extract_validity_from_back(lines: Sequence[str], text: str) -> Tuple[Optional[str], Optional[str]]:
    raw_lines = [x.strip() for x in lines if x and x.strip()]
    merged = normalize_text(text)
    back_context = sum(1 for k in BACK_KEYWORDS if k in merged) >= 2 or '有效期限' in merged or '签发机关' in merged
    if not back_context:
        return None, None

    candidates = []

    def scan_one(piece: str, base_score: float) -> None:
        s = normalize_text(piece)
        if not s:
            return
        score = base_score + (40 if ('有效期限' in s or '有效期' in s) else 0) + (5 if '签发机关' in s else 0)
        for pat in [
            r'(\d{4}[./-]\d{1,2}[./-]\d{1,2})\s*[至-]\s*(\d{4}[./-]\d{1,2}[./-]\d{1,2}|长期)',
            r'(\d{8})\s*[至-]\s*(\d{8}|长期)',
        ]:
            for m in re.finditer(pat, s):
                left = parse_std_date(m.group(1))
                right = '长期' if m.group(2) == '长期' else parse_std_date(m.group(2))
                if left and right and (right == '长期' or is_plausible_validity(left, right)):
                    candidates.append((left, right, score + 30 + (10 if m.group(0) == s else 0)))

        dates = re.findall(r'\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{8}', s)
        parsed = [parse_std_date(x) for x in dates]
        parsed = [x for x in parsed if x]
        if len(parsed) >= 2 and is_plausible_validity(parsed[0], parsed[1]):
            candidates.append((parsed[0], parsed[1], score + 10))
        elif len(parsed) == 1 and '长期' in s:
            candidates.append((parsed[0], '长期', score + 8))

    for line in raw_lines:
        scan_one(line, 20.0)
    scan_one(text, 5.0)

    if not candidates:
        return None, None
    left, right, _ = max(candidates, key=lambda x: x[2])
    return left, right
