from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from core.interfaces import ClassifierMethod


_DEFAULT_MODEL_NAME = "mobilenet_v3_small.tflite"
_DEFAULT_MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite",
    "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite",
]


def _extract_embedding_vector(embedding_obj) -> np.ndarray | None:
    """Compatibilidad entre variantes de API de MediaPipe Embedding."""
    if embedding_obj is None:
        return None

    for attr in ("feature_vector", "embedding", "float_embedding", "quantized_embedding"):
        value = getattr(embedding_obj, attr, None)
        if value is None:
            continue
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
        if vec.size > 0:
            return vec
    return None


def _ensure_model(path_or_none: str | None) -> str:
    if path_or_none:
        return path_or_none

    models_root = Path(__file__).resolve().parent.parent / "models"
    embed_dir = models_root / "embed"
    embed_dir.mkdir(parents=True, exist_ok=True)

    model_path = embed_dir / _DEFAULT_MODEL_NAME
    legacy_path = models_root / _DEFAULT_MODEL_NAME

    if model_path.exists():
        return str(model_path)
    if legacy_path.exists():
        return str(legacy_path)

    for url in _DEFAULT_MODEL_URLS:
        try:
            urllib.request.urlretrieve(url, str(model_path))
            return str(model_path)
        except Exception:
            continue

    raise RuntimeError(
        "No se pudo descargar el modelo de embedding de MediaPipe. "
        "Usa --embedder-model con una ruta valida."
    )


class MediaPipeEmbeddingMethod(ClassifierMethod):
    name = "mediapipe_embedding"

    def __init__(self, model_path: str | None = None):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception as exc:
            raise RuntimeError("Falta mediapipe. Instala dependencias de 04_Clasificador/requirements.txt") from exc

        self.mp = mp
        self.model_path = _ensure_model(model_path)
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True, quantize=False)
        self.embedder = vision.ImageEmbedder.create_from_options(options)

    def build_descriptor(self, image_bgr):
        if image_bgr is None or image_bgr.size == 0:
            return None

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.embedder.embed(mp_image)
        if not result.embeddings:
            return None

        vec = _extract_embedding_vector(result.embeddings[0])
        if vec is None:
            return None
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-8:
            return None
        return vec / norm

    def score(self, query_descriptor, model_descriptor) -> float:
        q = np.asarray(query_descriptor, dtype=np.float32)
        m = np.asarray(model_descriptor, dtype=np.float32)
        return float(np.dot(q, m))

    def format_score(self, value: float) -> str:
        return f"cos={value:.3f}"

