#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UMUCV_PKG = ROOT / "umucv" / "package"
if str(UMUCV_PKG) not in sys.path:
    sys.path.insert(0, str(UMUCV_PKG))


def add_stream_args(parser: argparse.ArgumentParser) -> None:
    try:
        from umucv.stream import sourceArgs  # type: ignore

        sourceArgs(parser)
    except Exception:
        parser.add_argument("--dev", default="0", help="fuente de video")


def get_stream(args: argparse.Namespace):
    try:
        from umucv.stream import autoStream  # type: ignore

        return autoStream()
    except Exception:
        src = args.dev
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir --dev={args.dev}")

        def fallback():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                key = cv.waitKey(1) & 0xFF
                yield key, frame
            cap.release()

        return fallback()


def gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return cv.GaussianBlur(x, (0, 0), sigma)


def normalize_response(r: np.ndarray) -> np.ndarray:
    r = r.astype(np.float32)
    rmin = float(np.min(r))
    rmax = float(np.max(r))
    if abs(rmax - rmin) < 1e-9:
        return np.zeros_like(r, dtype=np.float32)
    return (r - rmin) / (rmax - rmin)


def nms_points(response_norm: np.ndarray, quality: float, max_corners: int, nms_size: int) -> np.ndarray:
    nms_size = max(3, int(nms_size) | 1)
    kernel = np.ones((nms_size, nms_size), np.uint8)
    local_max = cv.dilate(response_norm, kernel)
    mask = (response_norm == local_max) & (response_norm >= float(quality))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)

    scores = response_norm[ys, xs]
    order = np.argsort(scores)[::-1]
    if max_corners > 0:
        order = order[:max_corners]
    pts = np.column_stack([xs[order], ys[order]]).astype(np.float32)
    return pts


def harris_manual(gray_f32: np.ndarray, sigma_grad: float, sigma_window: float, kappa: float) -> np.ndarray:
    g = gaussian(gray_f32, sigma_grad)
    ix = cv.Sobel(g, cv.CV_32F, 1, 0, ksize=3)
    iy = cv.Sobel(g, cv.CV_32F, 0, 1, ksize=3)

    sxx = gaussian(ix * ix, sigma_window)
    syy = gaussian(iy * iy, sigma_window)
    sxy = gaussian(ix * iy, sigma_window)

    det = sxx * syy - sxy * sxy
    trace = sxx + syy
    return det - kappa * trace * trace


def harris_opencv(gray_f32: np.ndarray, block_size: int, ksize: int, kappa: float) -> np.ndarray:
    return cv.cornerHarris(gray_f32, blockSize=block_size, ksize=ksize, k=kappa)


def match_ratio(points_a: np.ndarray, points_b: np.ndarray, radius: float) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    r2 = float(radius * radius)
    matched = 0
    for pa in points_a:
        d2 = np.sum((points_b - pa[None, :]) ** 2, axis=1)
        if float(np.min(d2)) <= r2:
            matched += 1
    return matched / float(max(1, len(points_a)))


def draw_points(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    for x, y in np.round(points).astype(np.int32):
        cv.circle(frame, (x, y), 3, color, -1, cv.LINE_AA)


def response_to_bgr(response_norm: np.ndarray) -> np.ndarray:
    vis = np.clip(response_norm * 255.0, 0, 255).astype(np.uint8)
    return cv.cvtColor(vis, cv.COLOR_GRAY2BGR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparacion entre detector Harris propio y OpenCV")
    add_stream_args(parser)
    parser.add_argument("--sigma-grad", type=float, default=1.2, help="suavizado previo al gradiente")
    parser.add_argument("--sigma-window", type=float, default=2.5, help="suavizado de la matriz de estructura")
    parser.add_argument("--kappa", type=float, default=0.04, help="constante Harris")
    parser.add_argument("--quality", type=float, default=0.15, help="umbral relativo 0..1 tras normalizacion")
    parser.add_argument("--max-corners", type=int, default=250, help="maximo de esquinas a mostrar")
    parser.add_argument("--nms-size", type=int, default=5, help="tamano de supresion de no maximos")
    parser.add_argument("--match-radius", type=float, default=5.0, help="radio de coincidencia entre detectores")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for key, frame in get_stream(args):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        t0 = time.time()
        response_manual = harris_manual(
            gray_f32=gray,
            sigma_grad=float(args.sigma_grad),
            sigma_window=float(args.sigma_window),
            kappa=float(args.kappa),
        )
        t1 = time.time()
        response_cv = harris_opencv(
            gray_f32=gray,
            block_size=2,
            ksize=3,
            kappa=float(args.kappa),
        )
        t2 = time.time()

        manual_norm = normalize_response(response_manual)
        cv_norm = normalize_response(response_cv)

        manual_pts = nms_points(
            response_norm=manual_norm,
            quality=float(args.quality),
            max_corners=int(args.max_corners),
            nms_size=int(args.nms_size),
        )
        cv_pts = nms_points(
            response_norm=cv_norm,
            quality=float(args.quality),
            max_corners=int(args.max_corners),
            nms_size=int(args.nms_size),
        )

        agree_manual = match_ratio(manual_pts, cv_pts, float(args.match_radius))
        agree_cv = match_ratio(cv_pts, manual_pts, float(args.match_radius))

        left = frame.copy()
        right = frame.copy()
        draw_points(left, manual_pts, (0, 0, 255))
        draw_points(right, cv_pts, (0, 255, 0))

        cv.putText(left, "Harris propio", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(right, "OpenCV cornerHarris", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

        top = np.hstack([left, right])
        bottom = np.hstack([response_to_bgr(manual_norm), response_to_bgr(cv_norm)])
        panel = np.vstack([top, bottom])

        h, w = panel.shape[:2]
        y0 = h - 90
        cv.putText(panel, f"manual: {len(manual_pts)} corners, {(t1 - t0)*1000:.1f} ms", (10, y0), cv.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv.LINE_AA)
        cv.putText(panel, f"opencv: {len(cv_pts)} corners, {(t2 - t1)*1000:.1f} ms", (10, y0 + 24), cv.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv.LINE_AA)
        cv.putText(panel, f"solape manual->opencv: {agree_manual*100:.1f}% | opencv->manual: {agree_cv*100:.1f}%", (10, y0 + 48), cv.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv.LINE_AA)
        cv.putText(panel, "Q o ESC sale", (10, y0 + 72), cv.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 1, cv.LINE_AA)

        cv.imshow("Harris Compare", panel)
        if key in (27, ord("q")):
            break

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
