#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UMUCV_PKG = ROOT / "umucv" / "package"
if str(UMUCV_PKG) not in sys.path:
    sys.path.insert(0, str(UMUCV_PKG))


@dataclass
class HandControlState:
    center_xy: np.ndarray
    hand_span_px: float
    angle_deg: float


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


def smooth_value(prev: np.ndarray | float | None, curr: np.ndarray | float, alpha: float):
    if prev is None:
        return curr
    return alpha * curr + (1.0 - alpha) * prev


def extract_hand_state(hand_landmarks, width: int, height: int) -> HandControlState:
    pts = np.array([[lm.x * width, lm.y * height] for lm in hand_landmarks.landmark], dtype=np.float32)

    wrist = pts[0]
    index_mcp = pts[5]
    pinky_mcp = pts[17]
    middle_mcp = pts[9]

    center_xy = np.mean(pts[[0, 5, 17]], axis=0)
    hand_span_px = float(np.linalg.norm(index_mcp - pinky_mcp))

    palm_axis = middle_mcp - wrist
    angle_deg = -math.degrees(math.atan2(float(palm_axis[1]), float(palm_axis[0])))

    return HandControlState(center_xy=center_xy, hand_span_px=hand_span_px, angle_deg=angle_deg)


def draw_virtual_object(
    canvas: np.ndarray,
    center_xy: np.ndarray,
    scale: float,
    angle_deg: float,
    color=(40, 220, 255),
) -> None:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    w = 110.0 * scale
    h = 70.0 * scale
    ang = math.radians(angle_deg)
    c, s = math.cos(ang), math.sin(ang)
    r = np.array([[c, -s], [s, c]], dtype=np.float32)

    rect = np.array(
        [
            [-w / 2, -h / 2],
            [w / 2, -h / 2],
            [w / 2, h / 2],
            [-w / 2, h / 2],
        ],
        dtype=np.float32,
    )
    rot_rect = rect @ r.T + np.array([cx, cy], dtype=np.float32)

    poly = np.round(rot_rect).astype(np.int32)
    cv.fillConvexPoly(canvas, poly, color, lineType=cv.LINE_AA)
    cv.polylines(canvas, [poly], True, (255, 255, 255), 2, cv.LINE_AA)

    arrow_len = 65.0 * scale
    tip = np.array([arrow_len, 0.0], dtype=np.float32) @ r.T + np.array([cx, cy], dtype=np.float32)
    center_int = (int(round(cx)), int(round(cy)))
    tip_int = tuple(np.round(tip).astype(np.int32))
    cv.arrowedLine(canvas, center_int, tip_int, (255, 255, 255), 3, cv.LINE_AA, tipLength=0.25)
    cv.circle(canvas, center_int, max(4, int(round(6 * scale))), (255, 255, 255), -1, cv.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlador sin contacto con mano y objeto virtual 2D")
    add_stream_args(parser)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True, help="modo espejo")
    parser.add_argument("--smooth", type=float, default=0.28, help="suavizado de control entre 0 y 1")
    parser.add_argument("--min-detect", type=float, default=0.5, help="confianza minima de deteccion")
    parser.add_argument("--min-track", type=float, default=0.5, help="confianza minima de tracking")
    parser.add_argument("--scale-ref", type=float, default=140.0, help="tamano base de mano en px para escala 1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import mediapipe as mp
    except Exception as exc:
        raise SystemExit("Falta mediapipe. Instala dependencias de las practicas con MediaPipe.") from exc

    if not hasattr(mp, "solutions"):
        venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
        hint = ""
        if venv_python.exists():
            hint = f"\nPrueba con:\n{venv_python} .\\08_Hand_Controller\\hand_controller.py --dev 0"
        raise SystemExit(
            "El interprete actual usa una variante de MediaPipe sin la API clasica 'mp.solutions'."
            " Este script necesita esa API.\n"
            f"Python actual: {sys.executable}{hint}"
        )

    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    smoothed_center = None
    smoothed_span = None
    smoothed_angle = None
    span_reference = float(args.scale_ref)
    fps_est = 25.0
    prev_t = None

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=1,
        min_detection_confidence=float(args.min_detect),
        min_tracking_confidence=float(args.min_track),
    ) as hands:
        for key, frame in get_stream(args):
            if bool(args.mirror):
                frame = cv.flip(frame, 1)

            now = time.time()
            if prev_t is None:
                prev_t = now
            dt = max(1e-3, now - prev_t)
            prev_t = now
            fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

            h, w = frame.shape[:2]
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            results = hands.process(rgb)

            vis = frame.copy()
            overlay = np.zeros_like(vis)

            if key == ord("r"):
                smoothed_center = None
                smoothed_span = None
                smoothed_angle = None

            if key == ord("b") and smoothed_span is not None:
                span_reference = float(smoothed_span)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                state = extract_hand_state(hand_landmarks, w, h)

                alpha = float(np.clip(args.smooth, 0.01, 1.0))
                smoothed_center = smooth_value(smoothed_center, state.center_xy, alpha)
                smoothed_span = float(smooth_value(smoothed_span, state.hand_span_px, alpha))
                smoothed_angle = float(smooth_value(smoothed_angle, state.angle_deg, alpha))

                scale = float(np.clip(smoothed_span / max(1.0, span_reference), 0.45, 2.8))

                draw_virtual_object(
                    overlay,
                    center_xy=np.asarray(smoothed_center, dtype=np.float32),
                    scale=scale,
                    angle_deg=smoothed_angle,
                )

                mp_drawing.draw_landmarks(
                    vis,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

                center_int = tuple(np.round(smoothed_center).astype(np.int32))
                cv.circle(vis, center_int, 7, (0, 255, 255), -1, cv.LINE_AA)
                cv.putText(vis, f"center=({center_int[0]}, {center_int[1]})", (10, 82), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA)
                cv.putText(vis, f"span={smoothed_span:.1f}px  scale={scale:.2f}", (10, 106), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA)
                cv.putText(vis, f"angle={smoothed_angle:+.1f} deg", (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA)
                cv.putText(vis, f"base-span={span_reference:.1f}px", (10, 154), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA)
            else:
                cv.putText(vis, "No hand detected", (10, 82), cv.FONT_HERSHEY_SIMPLEX, 0.6, (80, 160, 255), 2, cv.LINE_AA)

            out = cv.addWeighted(vis, 1.0, overlay, 0.75, 0.0)
            cv.putText(out, f"fps={fps_est:.1f}", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(out, "Move hand = move object", (10, 48), cv.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(out, "Bring hand closer = scale up | Rotate hand = rotate object", (10, h - 36), cv.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv.LINE_AA)
            cv.putText(out, "B set base span | R reset smoothing | Q or ESC quit", (10, h - 14), cv.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv.LINE_AA)

            cv.imshow("Hand Controller", out)
            if key in (27, ord("q")):
                break

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
