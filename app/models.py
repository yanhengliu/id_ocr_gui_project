from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OcrLine:
    text: str
    score: float


@dataclass
class OcrAttempt:
    tag: str
    lines: List[OcrLine] = field(default_factory=list)


@dataclass
class ImageAnalysis:
    image_path: str
    label: str = 'other'
    label_reason: str = ''
    front_score: float = 0.0
    back_score: float = 0.0
    portrait_score: float = 0.0
    other_score: float = 0.0
    front_role_score: float = 0.0
    back_role_score: float = 0.0
    portrait_role_score: float = 0.0
    front_hits: int = 0
    back_hits: int = 0
    bank_hits: int = 0
    face_count: int = 0
    largest_face_ratio: float = 0.0
    second_face_ratio: float = 0.0
    best_card_area_ratio: float = 0.0
    card_candidate_count: int = 0
    has_id_number: bool = False
    extracted_name: Optional[str] = None
    extracted_id: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    merged_text: str = ''
    merged_lines: List[str] = field(default_factory=list)
    ocr_attempt_tags: List[str] = field(default_factory=list)


@dataclass
class FolderResult:
    original_folder: str
    output_folder: str
    validity_bucket: str
    name: str
    id_number: str
    valid_from: str
    valid_to: str
    front_image: Optional[str]
    back_image: Optional[str]
    portrait_image: Optional[str]
    other_images: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
