from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import numpy.core.multiarray
import cv2

from .constants import BACK_KEYWORDS, BANK_KEYWORDS, FRONT_KEYWORDS, LABEL_BACK, LABEL_FRONT, LABEL_OTHER, LABEL_PORTRAIT
from .extractors import extract_name_strong, extract_validity_from_back
from .image_ops import preprocess_variants, read_image_unicode, rotate_bound
from .models import ImageAnalysis, OcrAttempt
from .ocr_wrapper import OCRWrapper
from .utils import count_hits, dedup_lines, extract_id_number, normalize_text


class ImageAnalyzer:
    def __init__(self, ocr: OCRWrapper, logger):
        self.ocr = ocr
        self.logger = logger
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def _iou(self, a, b) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _nms_boxes(self, boxes, iou_thr=0.35):
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        for box in boxes:
            if all(self._iou(box, k) < iou_thr for k in kept):
                kept.append(box)
        return kept

    def detect_faces_robust(self, image):
        h, w = image.shape[:2]
        img_area = h * w
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = min(h, w)
        boxes = []
        for min_scale, neighbors in [(0.04, 5), (0.06, 4), (0.08, 3)]:
            min_sz = max(24, int(min_side * min_scale))
            det = self.face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=neighbors, minSize=(min_sz, min_sz))
            for (x, y, fw, fh) in det:
                ratio = (fw * fh) / max(1.0, img_area)
                if ratio < 0.001 or ratio > 0.28:
                    continue
                ar = fw / max(1.0, fh)
                if ar < 0.65 or ar > 1.45:
                    continue
                boxes.append((int(x), int(y), int(fw), int(fh)))
        boxes = self._nms_boxes(boxes, 0.30)
        ratios = sorted([(bw * bh) / max(1.0, img_area) for (_, _, bw, bh) in boxes], reverse=True)
        largest = ratios[0] if ratios else 0.0
        second = ratios[1] if len(ratios) > 1 else 0.0
        return len(ratios), float(largest), float(second)

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        tl, tr, br, bl = rect
        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = int(max(width_a, width_b))
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = int(max(height_a, height_b))
        if max_width < 10 or max_height < 10:
            return np.empty((0, 0, 3), dtype=np.uint8)
        dst = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype=np.float32)
        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, m, (max_width, max_height))
        if warped.shape[0] > warped.shape[1]:
            warped = rotate_bound(warped, 90)
        return warped

    def detect_card_candidates(self, image, topk=4):
        h, w = image.shape[:2]
        img_area = h * w
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        masks = []
        edges = cv2.Canny(gray, 50, 150)
        masks.append(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))
        _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(cv2.morphologyEx(th1, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)))

        all_cands = []
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img_area * 0.015:
                    continue
                rect = cv2.minAreaRect(cnt)
                (_, _), (rw, rh), _ = rect
                if rw <= 1 or rh <= 1:
                    continue
                ratio = max(rw, rh) / max(1.0, min(rw, rh))
                if ratio < 1.2 or ratio > 2.4:
                    continue
                area_ratio = area / max(1.0, img_area)
                if area_ratio > 0.95:
                    continue
                box = cv2.boxPoints(rect).astype(np.float32)
                crop = self.four_point_transform(image, box)
                if crop.size == 0:
                    continue
                all_cands.append((crop, float(area_ratio)))
        all_cands.sort(key=lambda x: x[1], reverse=True)
        uniq, seen = [], []
        for crop, ratio in all_cands:
            ch, cw = crop.shape[:2]
            sig = (round(ratio * 1000), round(cw / 20), round(ch / 20))
            if sig in seen:
                continue
            seen.append(sig)
            uniq.append((crop, ratio))
            if len(uniq) >= topk:
                break
        return uniq

    def merge_attempts(self, attempts: Sequence[OcrAttempt]) -> Tuple[List[str], str]:
        merged = dedup_lines([line for a in attempts for line in a.lines])
        merged = sorted(merged, key=lambda z: (-z.score, len(normalize_text(z.text)), z.text))
        merged_lines = [x.text.strip() for x in merged]
        return merged_lines, ' '.join(merged_lines)

    def coarse_classify(self, merged_text, front_hits, back_hits, bank_hits, has_id, has_valid, card_area, card_count, face_count, largest_face_ratio, second_face_ratio):
        front_score = front_hits * 16.0
        back_score = back_hits * 18.0
        portrait_score = 0.0
        other_score = 0.0
        if has_id:
            front_score += 35
        if has_valid:
            back_score += 60
        if card_area > 0.22:
            front_score += 12
            back_score += 12
        if card_count > 0:
            front_score += 8
            back_score += 8
        if largest_face_ratio > 0.03:
            portrait_score += 15
        if largest_face_ratio > 0.055:
            portrait_score += 28
        if largest_face_ratio > 0.075:
            portrait_score += 38
        if second_face_ratio > 0.006:
            portrait_score += 18
        if second_face_ratio > 0.012:
            portrait_score += 18
        if 0.01 <= card_area <= 0.35:
            portrait_score += 12
        if has_id and largest_face_ratio > 0.06:
            portrait_score += 28
        if front_hits >= 2 and largest_face_ratio > 0.06:
            portrait_score += 15
        if bank_hits > 0 and front_hits == 0 and back_hits == 0:
            other_score += 80
        if has_valid:
            front_score -= 20
            portrait_score -= 35
        if back_hits >= 2:
            portrait_score -= 12
        scores = {LABEL_FRONT: front_score, LABEL_BACK: back_score, LABEL_PORTRAIT: portrait_score, LABEL_OTHER: other_score}
        label = max(scores, key=scores.get)
        reason = (
            f'front_hits={front_hits}, back_hits={back_hits}, bank_hits={bank_hits}, has_id={has_id}, '
            f'has_valid={has_valid}, card_area={card_area:.3f}, card_cands={card_count}, face_count={face_count}, '
            f'face_ratio={largest_face_ratio:.3f}, second_face_ratio={second_face_ratio:.3f}'
        )
        if scores[label] < 18:
            label = LABEL_OTHER
            reason += '; low-score -> other'
        return label, front_score, back_score, portrait_score, other_score, reason

    def compute_role_scores(self, a: ImageAnalysis):
        front_role = a.front_score
        if a.has_id_number:
            front_role += 18
        if a.extracted_name:
            front_role += 28
        if a.valid_from or a.valid_to:
            front_role -= 120
        if a.back_hits >= 2:
            front_role -= 80
        if a.largest_face_ratio > 0.075:
            front_role -= 65
        elif a.largest_face_ratio > 0.060:
            front_role -= 35
        elif 0.004 <= a.largest_face_ratio <= 0.055:
            front_role += 14
        if a.second_face_ratio > 0.010:
            front_role -= 18
        if a.best_card_area_ratio > 0.45:
            front_role += 12
        if 0.01 <= a.best_card_area_ratio <= 0.25:
            front_role -= 5

        back_role = a.back_score
        if a.valid_from and a.valid_to:
            back_role += 80
        if a.back_hits >= 2:
            back_role += 15
        if a.has_id_number:
            back_role -= 25
        if a.extracted_name:
            back_role -= 25
        if a.largest_face_ratio > 0.075:
            back_role -= 20

        portrait_role = a.portrait_score
        if a.valid_from or a.valid_to or a.back_hits >= 2:
            portrait_role -= 120
        if a.largest_face_ratio >= 0.085:
            portrait_role += 90
        elif a.largest_face_ratio >= 0.070:
            portrait_role += 65
        elif a.largest_face_ratio >= 0.055:
            portrait_role += 38
        elif a.largest_face_ratio >= 0.040:
            portrait_role += 14
        else:
            portrait_role -= 10
        if a.second_face_ratio >= 0.010:
            portrait_role += 32
        elif a.second_face_ratio >= 0.006:
            portrait_role += 16
        if 0.01 <= a.best_card_area_ratio <= 0.35:
            portrait_role += 18
        elif a.best_card_area_ratio > 0.45:
            portrait_role -= 25
        if a.has_id_number:
            portrait_role += 15
        if a.front_hits >= 2:
            portrait_role += 12
        if a.extracted_name:
            portrait_role += 8
        if a.largest_face_ratio < 0.030:
            portrait_role -= 20
        return float(front_role), float(back_role), float(portrait_role)

    def analyze(self, image_path: Path) -> ImageAnalysis:
        img = read_image_unicode(image_path)
        attempts = []
        for tag, variant in preprocess_variants(img):
            try:
                attempts.append(self.ocr.ocr_image(variant, f'whole_{tag}'))
            except Exception as e:
                self.logger.warning('OCR失败 whole %s tag=%s err=%s', image_path.name, tag, e)
        card_cands = self.detect_card_candidates(img, topk=4)
        best_card_area = 0.0
        for idx, (crop, area_ratio) in enumerate(card_cands):
            best_card_area = max(best_card_area, area_ratio)
            for tag, variant in preprocess_variants(crop):
                try:
                    attempts.append(self.ocr.ocr_image(variant, f'card{idx}_{tag}'))
                except Exception as e:
                    self.logger.warning('OCR失败 crop %s cand=%d tag=%s err=%s', image_path.name, idx, tag, e)
        merged_lines, merged_text = self.merge_attempts(attempts)
        merged_norm = normalize_text(merged_text)
        front_hits = count_hits(merged_norm, FRONT_KEYWORDS)
        back_hits = count_hits(merged_norm, BACK_KEYWORDS)
        bank_hits = count_hits(merged_norm, BANK_KEYWORDS)
        has_id = extract_id_number(merged_text) is not None
        v_from, v_to = extract_validity_from_back(merged_lines, merged_text)
        has_valid = bool(v_from and v_to)
        face_count, largest_face_ratio, second_face_ratio = self.detect_faces_robust(img)
        label, fs, bs, ps, os, reason = self.coarse_classify(merged_text, front_hits, back_hits, bank_hits, has_id, has_valid, best_card_area, len(card_cands), face_count, largest_face_ratio, second_face_ratio)
        name = extract_name_strong(merged_lines, merged_text) if front_hits >= 2 else None
        id_no = extract_id_number(merged_text) if (front_hits >= 1 or label in {LABEL_FRONT, LABEL_PORTRAIT}) else None
        analysis = ImageAnalysis(
            image_path=str(image_path), label=label, label_reason=reason,
            front_score=fs, back_score=bs, portrait_score=ps, other_score=os,
            front_hits=front_hits, back_hits=back_hits, bank_hits=bank_hits,
            face_count=face_count, largest_face_ratio=largest_face_ratio, second_face_ratio=second_face_ratio,
            best_card_area_ratio=best_card_area, card_candidate_count=len(card_cands), has_id_number=has_id,
            extracted_name=name, extracted_id=id_no, valid_from=v_from, valid_to=v_to,
            merged_text=merged_text, merged_lines=merged_lines, ocr_attempt_tags=[a.tag for a in attempts]
        )
        analysis.front_role_score, analysis.back_role_score, analysis.portrait_role_score = self.compute_role_scores(analysis)
        self.logger.info(
            'IMG=%s label=%s F=%.1f B=%.1f P=%.1f | roleF=%.1f roleB=%.1f roleP=%.1f | front_hits=%d back_hits=%d face=%.3f second=%.3f card=%.3f name=%s id=%s valid=%s~%s',
            image_path.name, analysis.label, analysis.front_score, analysis.back_score, analysis.portrait_score,
            analysis.front_role_score, analysis.back_role_score, analysis.portrait_role_score,
            analysis.front_hits, analysis.back_hits, analysis.largest_face_ratio, analysis.second_face_ratio,
            analysis.best_card_area_ratio, analysis.extracted_name, analysis.extracted_id, analysis.valid_from, analysis.valid_to,
        )
        return analysis
