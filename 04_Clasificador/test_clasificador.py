import unittest
import os
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from methods.hand_procrustes import procrustes_distance
from methods.mediapipe_embedding import _extract_embedding_vector
from methods.sift_matching import SIFTMatchingMethod


class TestClasificador(unittest.TestCase):
    def test_mediapipe_embedding_extract_vector_compat(self):
        class EmbFeature:
            def __init__(self):
                self.feature_vector = [0.1, 0.2, 0.3]

        class EmbEmbedding:
            def __init__(self):
                self.embedding = [1.0, 2.0]

        v1 = _extract_embedding_vector(EmbFeature())
        v2 = _extract_embedding_vector(EmbEmbedding())

        self.assertIsNotNone(v1)
        self.assertIsNotNone(v2)
        self.assertEqual(tuple(v1.shape), (3,))
        self.assertEqual(tuple(v2.shape), (2,))

    def test_procrustes_invariante_a_traslacion_y_escala(self):
        a = np.array([[0.0, 0.0], [1.0, 0.2], [0.5, 0.8], [0.2, 0.5]], dtype=np.float32)
        b = a * 2.5 + np.array([7.0, -3.0], dtype=np.float32)
        d = procrustes_distance(a, b)
        self.assertLess(d, 1e-4)

    def test_sift_reconoce_patron_parecido(self):
        try:
            method = SIFTMatchingMethod()
        except RuntimeError:
            self.skipTest("SIFT no disponible en esta instalacion")
            return

        base = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(base, "BOOK", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        cv2.circle(base, (250, 70), 30, (255, 255, 255), 2)

        similar = cv2.GaussianBlur(base, (3, 3), 0)
        different = np.zeros_like(base)
        cv2.rectangle(different, (20, 20), (300, 220), (255, 255, 255), -1)

        d_base = method.build_descriptor(base)
        d_sim = method.build_descriptor(similar)
        d_diff = method.build_descriptor(different)

        self.assertIsNotNone(d_base)
        self.assertIsNotNone(d_sim)
        if d_diff is None:
            # Si no hay keypoints suficientes en la imagen "different", ya es evidencia de no coincidencia.
            return

        s1 = method.score(d_base, d_sim)
        s2 = method.score(d_base, d_diff)
        self.assertGreater(s1, s2)


if __name__ == "__main__":
    unittest.main()

