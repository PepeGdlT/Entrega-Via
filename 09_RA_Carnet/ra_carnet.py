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


CARD_RATIO = 85.60 / 53.98
CANONICAL_W = 960
CANONICAL_H = int(round(CANONICAL_W / CARD_RATIO))
CARD_CANONICAL = np.array(
    [[0.0, 0.0], [CANONICAL_W - 1.0, 0.0], [CANONICAL_W - 1.0, CANONICAL_H - 1.0], [0.0, CANONICAL_H - 1.0]],
    dtype=np.float32,
)


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


def order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    out = np.zeros((4, 2), dtype=np.float32)
    out[0] = pts[np.argmin(s)]
    out[2] = pts[np.argmax(s)]
    out[1] = pts[np.argmin(d)]
    out[3] = pts[np.argmax(d)]
    return out


def quad_dims(quad: np.ndarray) -> tuple[float, float]:
    tl, tr, br, bl = quad
    w = 0.5 * (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl))
    h = 0.5 * (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr))
    return float(w), float(h)


def inside_frame(quad: np.ndarray, shape: tuple[int, int, int], margin: int = 4) -> bool:
    h, w = shape[:2]
    xs = quad[:, 0]
    ys = quad[:, 1]
    return xs.min() >= margin and ys.min() >= margin and xs.max() <= w - margin and ys.max() <= h - margin


def preprocess(gray: np.ndarray) -> np.ndarray:
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 60, 160)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel, iterations=2)
    return edges


def candidate_score(quad: np.ndarray, area: float, shape: tuple[int, int, int]) -> float:
    frame_area = float(shape[0] * shape[1])
    w, h = quad_dims(quad)
    if min(w, h) <= 1e-6:
        return -1e9
    ratio = max(w, h) / min(w, h)
    ratio_err = abs(ratio - CARD_RATIO)
    fill = area / frame_area
    rectangularity = area / max(1.0, w * h)
    center = np.mean(quad, axis=0)
    target = np.array([shape[1] / 2.0, shape[0] / 2.0], dtype=np.float32)
    center_penalty = np.linalg.norm(center - target) / max(shape[0], shape[1])
    return 4.0 * fill + 1.5 * rectangularity - 2.2 * ratio_err - 0.5 * center_penalty


def detect_card_quad(frame: np.ndarray, min_area_ratio: float) -> np.ndarray | None:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = preprocess(gray)
    contours = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    frame_area = float(frame.shape[0] * frame.shape[1])

    best_quad = None
    best_score = -1e9

    for contour in contours:
        area = abs(cv.contourArea(contour))
        if area < min_area_ratio * frame_area:
            continue
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4 or not cv.isContourConvex(approx):
            continue

        quad = order_quad(approx.reshape(4, 2))
        if not inside_frame(quad, frame.shape):
            continue

        w, h = quad_dims(quad)
        if min(w, h) < 80:
            continue

        ratio = max(w, h) / max(1e-6, min(w, h))
        if not (1.25 <= ratio <= 1.95):
            continue

        score = candidate_score(quad, area, frame.shape)
        if score > best_score:
            best_score = score
            best_quad = quad

    return best_quad


def draw_quad(frame: np.ndarray, quad: np.ndarray, color: tuple[int, int, int], label: str | None = None) -> None:
    cv.polylines(frame, [np.round(quad).astype(np.int32)], True, color, 2, cv.LINE_AA)
    if label:
        x, y = np.round(quad[0]).astype(int)
        cv.putText(frame, label, (x + 6, y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv.LINE_AA)


def warp_overlay(frame: np.ndarray, overlay: np.ndarray, dst_quad: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    src_quad = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype=np.float32)
    hmat = cv.getPerspectiveTransform(src_quad, dst_quad.astype(np.float32))

    warped = cv.warpPerspective(overlay, hmat, (w, h))
    mask = np.ones((oh, ow), dtype=np.uint8) * 255
    warped_mask = cv.warpPerspective(mask, hmat, (w, h))
    warped_mask = cv.GaussianBlur(warped_mask, (5, 5), 0)

    mask_f = warped_mask.astype(np.float32) / 255.0
    if warped.ndim == 3 and warped.shape[2] == 4:
        alpha = warped[:, :, 3].astype(np.float32) / 255.0
        mask_f *= alpha
        warped_rgb = warped[:, :, :3]
    else:
        warped_rgb = warped

    out = frame.astype(np.float32)
    for c in range(3):
        out[:, :, c] = out[:, :, c] * (1.0 - mask_f) + warped_rgb[:, :, c].astype(np.float32) * mask_f
    return np.clip(out, 0, 255).astype(np.uint8)


def rectify_card(frame: np.ndarray, card_quad: np.ndarray) -> np.ndarray:
    hmat = cv.getPerspectiveTransform(card_quad.astype(np.float32), CARD_CANONICAL.astype(np.float32))
    return cv.warpPerspective(frame, hmat, (CANONICAL_W, CANONICAL_H))


def roi_to_quad(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)


def quad_to_normalized(quad: np.ndarray) -> np.ndarray:
    norm = quad.copy().astype(np.float32)
    norm[:, 0] /= float(CANONICAL_W - 1)
    norm[:, 1] /= float(CANONICAL_H - 1)
    return norm


def normalized_to_image_quad(card_quad: np.ndarray, region_quad: np.ndarray) -> np.ndarray:
    region_px = region_quad.copy().astype(np.float32)
    region_px[:, 0] *= float(CANONICAL_W - 1)
    region_px[:, 1] *= float(CANONICAL_H - 1)
    hmat = cv.getPerspectiveTransform(CARD_CANONICAL.astype(np.float32), card_quad.astype(np.float32))
    return cv.perspectiveTransform(region_px.reshape(1, -1, 2), hmat).reshape(-1, 2)


def smooth_quad(prev_quad: np.ndarray | None, new_quad: np.ndarray, alpha: float) -> np.ndarray:
    if prev_quad is None:
        return new_quad.astype(np.float32)
    return ((1.0 - alpha) * prev_quad + alpha * new_quad).astype(np.float32)


def capture_template(stream, window_name: str, min_area_ratio: float, debug: bool) -> tuple[np.ndarray, np.ndarray]:
    print("Coloca el carnet frontal y pulsa T para capturar la plantilla.")
    print("Controles: T captura plantilla | Q o ESC sale")

    for key, frame in stream:
        vis = frame.copy()
        detected = detect_card_quad(frame, min_area_ratio)
        if detected is not None:
            draw_quad(vis, detected, (0, 255, 255), "carnet")

        cv.putText(vis, "T captura plantilla | Q o ESC sale", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.imshow(window_name, vis)

        if key in (27, ord("q")):
            raise SystemExit(0)
        if key == ord("t"):
            if detected is None:
                print("No se detecto el carnet. Prueba a acercarlo, frontal y con mejor contraste.")
                continue
            template_rectified = rectify_card(frame, detected)
            return frame.copy(), template_rectified

    raise SystemExit("No hay frames disponibles")


def select_photo_region(template_rectified: np.ndarray) -> np.ndarray:
    print("Selecciona la zona exacta de la foto sobre la plantilla rectificada.")
    print("Arrastra un rectangulo y confirma con ENTER o SPACE.")
    roi = cv.selectROI("Selecciona foto del carnet", template_rectified, showCrosshair=True, fromCenter=False)
    cv.destroyWindow("Selecciona foto del carnet")
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise SystemExit("No se selecciono ninguna region para la foto.")
    return quad_to_normalized(roi_to_quad(x, y, w, h))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sustitucion automatica de la foto del carnet con plantilla capturada desde la app")
    add_stream_args(parser)
    parser.add_argument("--replace", required=True, help="imagen a proyectar sobre la zona de la foto")
    parser.add_argument("--min-area-ratio", type=float, default=0.06, help="area minima del carnet respecto al frame")
    parser.add_argument("--smooth", type=float, default=0.35, help="suavizado temporal del cuadrilatero detectado")
    parser.add_argument("--hold", type=int, default=6, help="frames que se mantiene el ultimo carnet si se pierde deteccion")
    parser.add_argument("--debug", action="store_true", help="muestra el cuadrilatero del carnet y la zona detectada de la foto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overlay = cv.imread(args.replace, cv.IMREAD_UNCHANGED)
    if overlay is None:
        raise SystemExit(f"No se pudo abrir la imagen de reemplazo: {args.replace}")

    stream = iter(get_stream(args))
    window_name = "RA Carnet"
    cv.namedWindow(window_name)

    _, template_rectified = capture_template(stream, window_name, float(args.min_area_ratio), bool(args.debug))
    photo_region_norm = select_photo_region(template_rectified)

    smoothed_card_quad: np.ndarray | None = None
    missed = 0
    fps_est = 25.0
    prev_t = None

    print("Plantilla capturada. Ejecutando sustitucion automatica en tiempo real.")
    print("Controles: R reinicia el suavizado | Q o ESC sale")

    for key, frame in stream:
        now = time.time()
        if prev_t is None:
            prev_t = now
        dt = max(1e-3, now - prev_t)
        prev_t = now
        fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

        if key == ord("r"):
            smoothed_card_quad = None
            missed = 0

        vis = frame.copy()
        detected = detect_card_quad(frame, float(args.min_area_ratio))

        if detected is not None:
            smoothed_card_quad = smooth_quad(smoothed_card_quad, detected, float(args.smooth))
            missed = 0
        elif smoothed_card_quad is not None:
            missed += 1
            if missed > int(args.hold):
                smoothed_card_quad = None

        status = "SEARCH"
        if smoothed_card_quad is not None:
            photo_quad = normalized_to_image_quad(smoothed_card_quad, photo_region_norm)
            vis = warp_overlay(vis, overlay, photo_quad)
            status = "LOCK"
            if args.debug:
                draw_quad(vis, smoothed_card_quad, (0, 255, 255), "carnet")
                draw_quad(vis, photo_quad, (0, 165, 255), "foto")
        elif args.debug and detected is not None:
            draw_quad(vis, detected, (0, 255, 255), "carnet")

        cv.putText(vis, f"{status}  fps={fps_est:.1f}", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(vis, "R reinicia | Q o ESC sale", (10, frame.shape[0] - 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv.LINE_AA)
        cv.imshow(window_name, vis)

        if key in (27, ord("q")):
            break

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
