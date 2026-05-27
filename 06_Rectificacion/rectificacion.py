#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RefPoint:
    label: str
    world_xy: np.ndarray
    image_xy: np.ndarray


@dataclass
class MeasurePoint:
    label: str
    image_xy: np.ndarray


def order_polygon_clockwise(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)]


def parse_refs_file(path: Path) -> tuple[list[RefPoint], list[MeasurePoint], float]:
    ref_points: list[RefPoint] = []
    measure_points: list[MeasurePoint] = []
    scale = 4.0

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        kind = parts[0].upper()

        if kind == "SCALE":
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno}: SCALE requiere 1 valor")
            scale = float(parts[1])
            continue

        if kind == "REF":
            if len(parts) != 6:
                raise ValueError(f"{path}:{lineno}: REF requiere: REF label X Y x y")
            label = parts[1]
            world_xy = np.array([float(parts[2]), float(parts[3])], dtype=np.float32)
            image_xy = np.array([float(parts[4]), float(parts[5])], dtype=np.float32)
            ref_points.append(RefPoint(label=label, world_xy=world_xy, image_xy=image_xy))
            continue

        if kind == "MEASURE":
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: MEASURE requiere: MEASURE label x y")
            label = parts[1]
            image_xy = np.array([float(parts[2]), float(parts[3])], dtype=np.float32)
            measure_points.append(MeasurePoint(label=label, image_xy=image_xy))
            continue

        raise ValueError(f"{path}:{lineno}: tipo desconocido: {parts[0]}")

    if len(ref_points) < 4:
        raise ValueError(f"Se necesitan al menos 4 referencias REF, hay {len(ref_points)}")

    return ref_points, measure_points, scale


def compute_homography(ref_points: list[RefPoint]) -> np.ndarray:
    image_pts = np.array([p.image_xy for p in ref_points], dtype=np.float32)
    world_pts = np.array([p.world_xy for p in ref_points], dtype=np.float32)
    h, status = cv2.findHomography(image_pts, world_pts)
    if h is None or status is None:
        raise RuntimeError("No se pudo calcular la homografia")
    return h


def htrans(h: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    homog = np.hstack([x, np.ones((len(x), 1), dtype=np.float64)])
    proj = homog @ h.T
    return proj[:, :2] / proj[:, 2:3]


def build_rectification_canvas(
    image: np.ndarray,
    h_img_to_world: np.ndarray,
    pixels_per_unit: float,
    margin_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    image_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    world_corners = htrans(h_img_to_world, image_corners)
    min_xy = np.min(world_corners, axis=0)
    max_xy = np.max(world_corners, axis=0)

    width_px = int(math.ceil((max_xy[0] - min_xy[0]) * pixels_per_unit + 2 * margin_px))
    height_px = int(math.ceil((max_xy[1] - min_xy[1]) * pixels_per_unit + 2 * margin_px))
    width_px = max(100, width_px)
    height_px = max(100, height_px)

    t = np.array(
        [
            [pixels_per_unit, 0.0, margin_px - pixels_per_unit * min_xy[0]],
            [0.0, pixels_per_unit, margin_px - pixels_per_unit * min_xy[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rectif_h = t @ h_img_to_world
    rectified = cv2.warpPerspective(image, rectif_h, (width_px, height_px))
    return rectified, rectif_h


def draw_reference_overlay(image: np.ndarray, ref_points: list[RefPoint], measure_points: list[MeasurePoint]) -> np.ndarray:
    vis = image.copy()

    ref_poly = np.array([p.image_xy for p in ref_points], dtype=np.float32)
    if len(ref_poly) >= 4:
        poly = order_polygon_clockwise(ref_poly).astype(int)
        cv2.polylines(vis, [poly], True, (0, 255, 255), 2, cv2.LINE_AA)

    for p in ref_points:
        x, y = np.round(p.image_xy).astype(int)
        cv2.circle(vis, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        label = f"{p.label} ({p.world_xy[0]:.1f},{p.world_xy[1]:.1f})"
        cv2.putText(vis, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    for p in measure_points:
        x, y = np.round(p.image_xy).astype(int)
        cv2.circle(vis, (x, y), 5, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(vis, p.label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

    return vis


def annotate_measurement(
    image: np.ndarray,
    h_img_to_world: np.ndarray,
    p1_img: np.ndarray,
    p2_img: np.ndarray,
    units: str,
) -> tuple[np.ndarray, float, np.ndarray]:
    vis = image.copy()
    pair_img = np.array([p1_img, p2_img], dtype=np.float32)
    pair_world = htrans(h_img_to_world, pair_img).astype(np.float32)
    dist = float(np.linalg.norm(pair_world[1] - pair_world[0]))

    p1 = tuple(np.round(pair_img[0]).astype(int))
    p2 = tuple(np.round(pair_img[1]).astype(int))
    cv2.circle(vis, p1, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(vis, p2, 6, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(vis, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)

    mx = int(round((p1[0] + p2[0]) * 0.5))
    my = int(round((p1[1] + p2[1]) * 0.5))
    text = f"{dist:.2f} {units}"
    cv2.putText(vis, text, (mx + 8, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    return vis, dist, pair_world


class ClickMeasure:
    def __init__(self, window_name: str):
        self.window_name = window_name
        self.points: list[np.ndarray] = []
        self.last_pair: tuple[np.ndarray, np.ndarray] | None = None
        cv2.setMouseCallback(window_name, self.on_mouse)

    def on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) >= 2:
                self.points.clear()
            if not self.points and self.last_pair is not None:
                # Empezar una nueva medida reemplaza la anterior.
                self.last_pair = None
            self.points.append(np.array([x, y], dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rectificacion de un plano para medir distancias")
    parser.add_argument("--image", required=True, help="ruta de imagen")
    parser.add_argument("--refs", required=True, help="fichero de referencias")
    parser.add_argument("--units", default="mm", help="unidades reales del fichero de referencias")
    parser.add_argument("--margin", type=int, default=40, help="margen de la vista rectificada")
    parser.add_argument("--no-rectified", action="store_true", help="no mostrar la imagen rectificada")
    parser.add_argument("--save", default="", help="ruta opcional para guardar la imagen anotada")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    refs_path = Path(args.refs)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"No se pudo abrir la imagen: {image_path}")

    ref_points, measure_points, pixels_per_unit = parse_refs_file(refs_path)
    h_img_to_world = compute_homography(ref_points)
    rectified, rectif_h = build_rectification_canvas(
        image=image,
        h_img_to_world=h_img_to_world,
        pixels_per_unit=pixels_per_unit,
        margin_px=int(args.margin),
    )

    base_overlay = draw_reference_overlay(image, ref_points, measure_points)

    if len(measure_points) >= 2:
        measured_overlay, dist, pair_world = annotate_measurement(
            base_overlay,
            h_img_to_world,
            measure_points[0].image_xy,
            measure_points[1].image_xy,
            units=args.units,
        )
        print(f"[measure:file] distancia = {dist:.3f} {args.units}")
        print(f"[measure:file] P1 = {pair_world[0]}")
        print(f"[measure:file] P2 = {pair_world[1]}")
    else:
        measured_overlay = base_overlay

    window_name = "Rectificacion - original"
    cv2.namedWindow(window_name)
    click_measure = ClickMeasure(window_name)

    if not args.no_rectified:
        rectified_vis = rectified.copy()
        ref_img = np.array([p.image_xy for p in ref_points], dtype=np.float32)
        ref_rect = htrans(rectif_h, ref_img).astype(int)
        cv2.polylines(rectified_vis, [order_polygon_clockwise(ref_rect).astype(int)], True, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Rectificacion - rectificada", rectified_vis)

    print("Controles: click izq dos veces para medir, C limpia la medida interactiva, Q o ESC sale.")

    while True:
        vis = measured_overlay.copy()
        if click_measure.last_pair is not None:
            vis, _dist, _pair_world = annotate_measurement(
                vis,
                h_img_to_world,
                click_measure.last_pair[0],
                click_measure.last_pair[1],
                units=args.units,
            )
        if len(click_measure.points) == 1:
            p = tuple(np.round(click_measure.points[0]).astype(int))
            cv2.circle(vis, p, 6, (255, 0, 0), -1, cv2.LINE_AA)
        if len(click_measure.points) == 2:
            vis, dist, pair_world = annotate_measurement(
                vis,
                h_img_to_world,
                click_measure.points[0],
                click_measure.points[1],
                units=args.units,
            )
            print(f"[measure:click] distancia = {dist:.3f} {args.units}")
            print(f"[measure:click] P1 = {pair_world[0]}")
            print(f"[measure:click] P2 = {pair_world[1]}")
            click_measure.last_pair = (click_measure.points[0].copy(), click_measure.points[1].copy())
            click_measure.points.clear()

        cv2.putText(
            vis,
            "Click x2 mide | C limpia | Q sale",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            if args.save:
                cv2.imwrite(args.save, vis)
            break
        if key == ord("c"):
            click_measure.points.clear()
            click_measure.last_pair = None

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
