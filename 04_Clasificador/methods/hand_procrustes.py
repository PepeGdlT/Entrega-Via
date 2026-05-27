from __future__ import annotations

import cv2
import numpy as np

from core.interfaces import ClassifierMethod


def _normalize_points(points: np.ndarray) -> np.ndarray | None:
    if points is None or len(points) == 0:
        return None
    pts = np.asarray(points, dtype=np.float32)
    pts = pts - np.mean(pts, axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.sum(pts * pts, axis=1))))
    if scale <= 1e-8:
        return None
    return pts / scale


def procrustes_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    a = _normalize_points(points_a)
    b = _normalize_points(points_b)
    if a is None or b is None or a.shape != b.shape:
        return 1e9

    h = a.T @ b
    u, _s, vt = np.linalg.svd(h)
    r = u @ vt
    aligned = a @ r
    dist = np.linalg.norm(aligned - b) / np.sqrt(float(a.shape[0]))
    return float(dist)


class HandProcrustesMethod(ClassifierMethod):
    name = "hand_procrustes"

    def __init__(self):
        try:
            import mediapipe as mp
        except Exception as exc:
            raise RuntimeError("Falta mediapipe. Instala dependencias de 04_Clasificador/requirements.txt") from exc

        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def build_descriptor(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        out = self.hands.process(rgb)
        if not out.multi_hand_landmarks:
            return None

        hand = out.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y] for lm in hand.landmark], dtype=np.float32)
        return _normalize_points(pts)

    def score(self, query_descriptor, model_descriptor) -> float:
        d = procrustes_distance(query_descriptor, model_descriptor)
        if d >= 1e8:
            return -1.0
        return float(1.0 / (1.0 + d))

    def format_score(self, value: float) -> str:
        return f"proc={value:.3f}"

