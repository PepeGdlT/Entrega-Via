from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class ModelEntry:
    label: str
    path: str
    descriptor: object


@dataclass
class MatchHit:
    label: str
    model_path: str
    score: float


@dataclass
class ClassificationResult:
    best_hit: Optional[MatchHit]
    top_hits: List[MatchHit]
    label_scores: Dict[str, float]
    confidence: float


class ModelStore:
    def __init__(self, method_impl):
        self.method = method_impl
        self.entries: List[ModelEntry] = []

    def load_from_directory(self, models_dir: str) -> int:
        root = Path(models_dir)
        if not root.exists():
            return 0

        loaded = 0
        for file_path in sorted(root.rglob("*")):
            if file_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not file_path.is_file():
                continue

            label = file_path.parent.name
            image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            descriptor = self.method.build_descriptor(image)
            if descriptor is None:
                continue

            self.entries.append(ModelEntry(label=label, path=str(file_path), descriptor=descriptor))
            loaded += 1

        return loaded

    def add_model_image(self, image_bgr: np.ndarray, label: str, save_path: Optional[str] = None) -> bool:
        descriptor = self.method.build_descriptor(image_bgr)
        if descriptor is None:
            return False

        model_path = save_path if save_path else f"<runtime>/{label}"
        self.entries.append(ModelEntry(label=label, path=model_path, descriptor=descriptor))
        return True

    def classify(self, query_bgr: np.ndarray, topk: int = 5) -> ClassificationResult:
        query_desc = self.method.build_descriptor(query_bgr)
        if query_desc is None or not self.entries:
            return ClassificationResult(best_hit=None, top_hits=[], label_scores={}, confidence=0.0)

        hits: List[MatchHit] = []
        for entry in self.entries:
            value = float(self.method.score(query_desc, entry.descriptor))
            hits.append(MatchHit(label=entry.label, model_path=entry.path, score=value))

        hits.sort(key=lambda h: h.score, reverse=True)
        top_hits = hits[: max(1, int(topk))]

        label_scores: Dict[str, float] = {}
        for h in hits:
            prev = label_scores.get(h.label, -1e9)
            if h.score > prev:
                label_scores[h.label] = h.score

        sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
        confidence = 0.0
        if len(sorted_labels) >= 2:
            confidence = max(0.0, sorted_labels[0][1] - sorted_labels[1][1])
        elif len(sorted_labels) == 1:
            confidence = max(0.0, sorted_labels[0][1])

        best_hit = top_hits[0] if top_hits else None
        return ClassificationResult(best_hit=best_hit, top_hits=top_hits, label_scores=label_scores, confidence=confidence)

