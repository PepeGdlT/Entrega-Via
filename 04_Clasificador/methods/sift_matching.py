from __future__ import annotations

import cv2
import numpy as np

from core.interfaces import ClassifierMethod


class SIFTMatchingMethod(ClassifierMethod):
    name = "sift_matching"

    def __init__(self, ratio_test: float = 0.75):
        try:
            self.sift = cv2.SIFT_create()
        except Exception as exc:
            raise RuntimeError(
                "No se pudo crear SIFT. Instala opencv-contrib-python o una version de OpenCV con SIFT."
            ) from exc

        self.ratio_test = float(max(0.4, min(ratio_test, 0.95)))
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def build_descriptor(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        keypoints, desc = self.sift.detectAndCompute(gray, None)
        if desc is None or len(keypoints) < 8:
            return None

        return {
            "desc": desc,
            "n_keypoints": int(len(keypoints)),
        }

    def score(self, query_descriptor, model_descriptor) -> float:
        qd = query_descriptor["desc"]
        md = model_descriptor["desc"]
        if qd is None or md is None or len(qd) < 2 or len(md) < 2:
            return -1.0

        knn = self.matcher.knnMatch(qd, md, k=2)
        good = 0
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good += 1

        den = float(max(1, min(query_descriptor["n_keypoints"], model_descriptor["n_keypoints"])))
        return good / den

    def format_score(self, value: float) -> str:
        return f"sift={value:.3f}"

