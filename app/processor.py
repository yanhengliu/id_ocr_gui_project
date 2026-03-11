from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .constants import END_UNKNOWN, ID_UNKNOWN, IMAGE_EXTS, NAME_UNKNOWN, START_UNKNOWN
from .models import FolderResult, ImageAnalysis
from .utils import sanitize_name_for_fs, unique_path, validity_bucket

ProgressCallback = Callable[[int, int, str], None]


class FolderProcessor:
    def __init__(self, input_root: Path, output_root: Path, analyzer, logger, save_debug_json: bool = True, stop_flag: Optional[Callable[[], bool]] = None):
        self.input_root = input_root
        self.output_root = output_root
        self.analyzer = analyzer
        self.logger = logger
        self.save_debug_json = save_debug_json
        self.stop_flag = stop_flag or (lambda: False)
        self.validity_root = self.output_root / '身份证有效信息分类'
        self.debug_root = self.output_root / 'debug'
        self.logs_root = self.output_root / 'logs'
        self.summary_rows: List[FolderResult] = []
        for p in [self.validity_root, self.debug_root, self.logs_root]:
            p.mkdir(parents=True, exist_ok=True)

    def list_subfolders(self) -> List[Path]:
        return sorted([p for p in self.input_root.iterdir() if p.is_dir()], key=lambda x: x.name)

    def list_images(self, folder: Path) -> List[Path]:
        return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS], key=lambda x: x.name)

    def choose_back(self, analyses: Sequence[ImageAnalysis]):
        if not analyses:
            return None
        cands = sorted(analyses, key=lambda x: (x.back_role_score, bool(x.valid_from and x.valid_to), x.back_hits), reverse=True)
        best = cands[0]
        return best if best.back_role_score >= 70 else None

    def choose_portrait(self, analyses, used: set[str], notes: List[str]):
        remain = [x for x in analyses if x.image_path not in used]
        if not remain:
            return None
        cands = sorted(remain, key=lambda x: (x.portrait_role_score, x.largest_face_ratio, x.second_face_ratio), reverse=True)
        best = cands[0]
        if best.portrait_role_score >= 78 and (best.largest_face_ratio >= 0.060 or (best.largest_face_ratio >= 0.050 and (best.has_id_number or best.best_card_area_ratio >= 0.01 or best.second_face_ratio >= 0.006))):
            return best
        remain2 = [x for x in remain if x.back_role_score < 80]
        if remain2:
            best2 = sorted(remain2, key=lambda x: (x.portrait_role_score, x.largest_face_ratio), reverse=True)[0]
            if best2.portrait_role_score >= 45:
                notes.append('个人照片低置信回退选出')
                return best2
        return None

    def choose_front(self, analyses, used: set[str]):
        remain = [x for x in analyses if x.image_path not in used]
        if not remain:
            return None
        cands = sorted(remain, key=lambda x: (x.front_role_score, bool(x.extracted_name), bool(x.extracted_id), -x.largest_face_ratio), reverse=True)
        best = cands[0]
        return best if best.front_role_score >= 55 else None

    def refine_identity(self, front, back, analyses):
        name = None
        id_no = None
        valid_from = None
        valid_to = None
        front_pool = [front] if front else []
        front_pool.extend(sorted([x for x in analyses if x is not front], key=lambda x: (x.front_role_score, bool(x.extracted_name), bool(x.extracted_id)), reverse=True))
        for x in front_pool:
            if x.extracted_name:
                name = x.extracted_name
                break
        for x in front_pool:
            if x.extracted_id:
                id_no = x.extracted_id
                break
        back_pool = [back] if back else []
        back_pool.extend(sorted([x for x in analyses if x is not back], key=lambda x: (x.back_role_score, bool(x.valid_from and x.valid_to)), reverse=True))
        for x in back_pool:
            if x.valid_from and x.valid_to:
                valid_from, valid_to = x.valid_from, x.valid_to
                break
        return sanitize_name_for_fs(name or NAME_UNKNOWN), id_no or ID_UNKNOWN, valid_from or START_UNKNOWN, valid_to or END_UNKNOWN

    def copy_selected(self, src: Optional[Path], dst_folder: Path, stem: str) -> Optional[str]:
        if src is None:
            return None
        dst = dst_folder / f'{stem}{src.suffix.lower() or ".jpg"}'
        shutil.copy2(src, dst)
        return dst.name

    def process_one_folder(self, folder: Path) -> FolderResult:
        self.logger.info('开始处理文件夹：%s', folder)
        images = self.list_images(folder)
        notes: List[str] = []
        if not images:
            out = unique_path(self.validity_root / '未识别有效期' / f'{NAME_UNKNOWN}----{ID_UNKNOWN}----{START_UNKNOWN}----{END_UNKNOWN}')
            out.mkdir(parents=True, exist_ok=True)
            return FolderResult(str(folder), str(out), '未识别有效期', NAME_UNKNOWN, ID_UNKNOWN, START_UNKNOWN, END_UNKNOWN, None, None, None, [], ['文件夹内无图片'])

        analyses: List[ImageAnalysis] = []
        folder_debug_dir = self.debug_root / sanitize_name_for_fs(folder.name)
        folder_debug_dir.mkdir(parents=True, exist_ok=True)
        for img_path in images:
            analysis = self.analyzer.analyze(img_path)
            analyses.append(analysis)
            if self.save_debug_json:
                with open(folder_debug_dir / f'{img_path.stem}.json', 'w', encoding='utf-8') as f:
                    json.dump(asdict(analysis), f, ensure_ascii=False, indent=2)

        used: set[str] = set()
        back = self.choose_back(analyses)
        if back:
            used.add(back.image_path)
        else:
            notes.append('未找到明确的身份证国徽/有效期面')
        portrait = self.choose_portrait(analyses, used, notes)
        if portrait:
            used.add(portrait.image_path)
        else:
            notes.append('未找到明确的个人照片')
        front = self.choose_front(analyses, used)
        if front:
            used.add(front.image_path)
        else:
            remain = [x for x in analyses if x.image_path not in used]
            if remain:
                front = sorted(remain, key=lambda x: (x.front_role_score, bool(x.extracted_name), bool(x.extracted_id)), reverse=True)[0]
                used.add(front.image_path)
                notes.append('身份证人像页低置信回退选出')
            else:
                notes.append('未找到明确的身份证人像页')
        if front and portrait and portrait.front_role_score - front.front_role_score >= 35 and portrait.largest_face_ratio < 0.055 and front.largest_face_ratio > 0.070:
            front, portrait = portrait, front
            notes.append('根据角色分数交换 front/portrait')

        name, id_no, start, end = self.refine_identity(front, back, analyses)
        validity = validity_bucket(end)
        out = unique_path(self.validity_root / validity / f'{name}----{id_no}----{start}----{end}')
        out.mkdir(parents=True, exist_ok=True)
        front_name = self.copy_selected(Path(front.image_path) if front else None, out, '1')
        back_name = self.copy_selected(Path(back.image_path) if back else None, out, '2')
        portrait_name = self.copy_selected(Path(portrait.image_path) if portrait else None, out, '3')
        selected_paths = {x for x in [front.image_path if front else None, back.image_path if back else None, portrait.image_path if portrait else None] if x}
        other_names = []
        for img in images:
            if str(img) in selected_paths:
                continue
            shutil.copy2(img, out / img.name)
            other_names.append(img.name)
        for fp in folder.iterdir():
            if fp.is_file() and fp.suffix.lower() not in IMAGE_EXTS and not fp.name.startswith('.'):
                shutil.copy2(fp, out / fp.name)
                other_names.append(fp.name)
        manifest = {
            'original_folder': str(folder), 'output_folder': str(out), 'validity_bucket': validity,
            'name': name, 'id_number': id_no, 'valid_from': start, 'valid_to': end,
            'selected': {'front': asdict(front) if front else None, 'back': asdict(back) if back else None, 'portrait': asdict(portrait) if portrait else None},
            'all_images': [asdict(x) for x in analyses], 'notes': notes,
        }
        with open(out / 'result.json', 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return FolderResult(str(folder), str(out), validity, name, id_no, start, end, front_name, back_name, portrait_name, other_names, notes)

    def save_summary(self):
        csv_path = self.logs_root / 'summary.csv'
        json_path = self.logs_root / 'summary.json'
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['original_folder', 'output_folder', 'validity_bucket', 'name', 'id_number', 'valid_from', 'valid_to', 'front_image', 'back_image', 'portrait_image', 'other_images', 'notes'])
            for row in self.summary_rows:
                writer.writerow([row.original_folder, row.output_folder, row.validity_bucket, row.name, row.id_number, row.valid_from, row.valid_to, row.front_image, row.back_image, row.portrait_image, ' | '.join(row.other_images), ' | '.join(row.notes)])
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([asdict(x) for x in self.summary_rows], f, ensure_ascii=False, indent=2)

    def run(self, progress_callback: Optional[ProgressCallback] = None):
        folders = self.list_subfolders()
        total = len(folders)
        self.logger.info('共发现 %d 个子文件夹', total)
        for idx, folder in enumerate(folders, 1):
            if self.stop_flag():
                self.logger.warning('用户请求停止，任务中断。')
                break
            msg = f'[{idx}/{total}] 处理 {folder.name}'
            self.logger.info(msg)
            if progress_callback:
                progress_callback(idx - 1, total, msg)
            result = self.process_one_folder(folder)
            self.summary_rows.append(result)
            if progress_callback:
                progress_callback(idx, total, f'完成 {folder.name}')
        self.save_summary()
        self.logger.info('全部处理完成。输出目录：%s', self.output_root)
