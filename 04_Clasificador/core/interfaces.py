from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class ClassifierMethod(ABC):
    """Interfaz comun para todos los metodos de comparacion."""

    name: str = "base"

    @abstractmethod
    def build_descriptor(self, image_bgr) -> Optional[Any]:
        """Extrae el descriptor de una imagen BGR. Devuelve None si no aplica."""

    @abstractmethod
    def score(self, query_descriptor: Any, model_descriptor: Any) -> float:
        """Devuelve similitud. Mayor es mejor."""

    def format_score(self, value: float) -> str:
        return f"{value:.3f}"

