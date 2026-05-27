#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class WorldRef:
    label: str
    x: float
    y: float


def parse_world_points(text: str, labels_text: str | None) -> list[WorldRef]:
    chunks = [c.strip() for c in text.split(";") if c.strip()]
    labels = [c.strip() for c in labels_text.split(",")] if labels_text else []
    refs: list[WorldRef] = []
    for idx, chunk in enumerate(chunks):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Punto invalido: {chunk}")
        label = labels[idx] if idx < len(labels) and labels[idx] else f"p{idx + 1}"
        refs.append(WorldRef(label=label, x=float(parts[0]), y=float(parts[1])))
    if len(refs) < 4:
        raise ValueError("Se necesitan al menos 4 puntos reales")
    return refs


class Picker:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.points: list[np.ndarray] = []
        cv2.setMouseCallback(window_name, self.on_mouse)

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(np.array([x, y], dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera un fichero REF haciendo click sobre la imagen")
    parser.add_argument("--image", required=True, help="imagen donde marcar las referencias")
    parser.add_argument("--world", required=True, help="lista de puntos reales: X1,Y1;X2,Y2;...")
    parser.add_argument("--labels", default="", help="etiquetas opcionales separadas por comas")
    parser.add_argument("--scale", type=float, default=4.0, help="pixeles por unidad para la vista rectificada")
    parser.add_argument("--output", default="", help="ruta opcional para guardar el fichero")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"No se pudo abrir la imagen: {image_path}")

    refs = parse_world_points(args.world, args.labels if args.labels else None)
    window_name = "Picker REF"
    cv2.namedWindow(window_name)
    picker = Picker(window_name)
    info_printed = False

    while True:
        vis = image.copy()
        for idx, point in enumerate(picker.points):
            x, y = np.round(point).astype(int)
            cv2.circle(vis, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
            if idx < len(refs):
                cv2.putText(vis, refs[idx].label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        if not info_printed:
            print("Haz click en los puntos de referencia en el mismo orden que --world.")
            print("Controles: U deshace, Q o ESC sale.")
            info_printed = True

        idx_next = min(len(picker.points), len(refs) - 1)
        next_ref = refs[idx_next]
        cv2.putText(
            vis,
            f"Siguiente: {next_ref.label} -> ({next_ref.x:.1f}, {next_ref.y:.1f})",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and picker.points:
            picker.points.pop()
        if key in (27, ord("q")):
            break
        if len(picker.points) == len(refs):
            lines = [f"SCALE {args.scale:g}"]
            for ref, img_xy in zip(refs, picker.points):
                lines.append(f"REF {ref.label} {ref.x:g} {ref.y:g} {img_xy[0]:.1f} {img_xy[1]:.1f}")
            text = "\n".join(lines)
            print("\n" + text + "\n")
            if args.output:
                Path(args.output).write_text(text + "\n", encoding="utf-8")
                print(f"Guardado en: {args.output}")
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
