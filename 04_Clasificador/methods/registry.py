from __future__ import annotations

from typing import Dict, List

from methods.hand_procrustes import HandProcrustesMethod
from methods.mediapipe_embedding import MediaPipeEmbeddingMethod
from methods.sift_matching import SIFTMatchingMethod


def available_methods() -> List[str]:
    return ["mediapipe_embedding", "hand_procrustes", "sift_matching"]


def create_method(method_name: str, args):
    method_name = (method_name or "").strip().lower()
    if method_name == "mediapipe_embedding":
        return MediaPipeEmbeddingMethod(model_path=args.embedder_model)
    if method_name == "hand_procrustes":
        return HandProcrustesMethod()
    if method_name == "sift_matching":
        return SIFTMatchingMethod(ratio_test=args.sift_ratio)
    raise ValueError(f"Metodo desconocido: {method_name}")

