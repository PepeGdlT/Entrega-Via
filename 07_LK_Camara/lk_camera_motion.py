#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2 as cv
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UMUCV_PKG = ROOT / "umucv" / "package"
if str(UMUCV_PKG) not in sys.path:
    sys.path.insert(0, str(UMUCV_PKG))


TRACK_LEN = 20
DETECT_INTERVAL = 5

CORNERS_PARAMS = dict(
    maxCorners=500,
    qualityLevel=0.1,
    minDistance=10,
    blockSize=7,
)

LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03),
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


def load_calibration(path: str | None) -> tuple[float | None, float | None]:
    if not path:
        return None, None
    raw = np.loadtxt(path).astype(np.float64).flatten()
    if raw.size < 9:
        raise ValueError(f"Calibracion invalida: {path}")
    k = raw[:9].reshape(3, 3)
    return float(k[0, 0]), float(k[1, 1])


def robust_center_flow(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flow = p1 - p0
    if len(flow) == 0:
        return np.zeros(2, dtype=np.float32), flow
    median = np.median(flow, axis=0)
    mad = np.median(np.abs(flow - median), axis=0) + 1e-6
    keep = np.all(np.abs(flow - median) <= 3.5 * mad, axis=1)
    kept = flow[keep]
    if len(kept) < max(8, len(flow) // 5):
        kept = flow
    return np.mean(kept, axis=0), kept


def robust_radial_change(p0: np.ndarray, p1: np.ndarray, center_xy: np.ndarray) -> float:
    if len(p0) == 0:
        return 0.0
    r0 = np.linalg.norm(p0 - center_xy[None, :], axis=1)
    r1 = np.linalg.norm(p1 - center_xy[None, :], axis=1)
    dr = r1 - r0
    return float(np.median(dr))


def estimate_deg_per_sec(dx_px: float, dy_px: float, dt: float, fx: float, fy: float) -> tuple[float, float, float]:
    yaw_deg = math.degrees(math.atan2(dx_px, max(1e-6, fx)))
    pitch_deg = math.degrees(math.atan2(dy_px, max(1e-6, fy)))
    yaw_rate = yaw_deg / max(dt, 1e-6)
    pitch_rate = pitch_deg / max(dt, 1e-6)
    total = math.hypot(yaw_rate, pitch_rate)
    return yaw_rate, pitch_rate, total


def classify_motion(
    dx: float,
    dy: float,
    dr: float,
    dx_thr: float,
    dy_thr: float,
    dr_thr: float,
) -> str:
    adx = abs(dx)
    ady = abs(dy)
    adr = abs(dr)

    if adr >= max(adx, ady) and adr >= dr_thr:
        return "FORWARD" if dr > 0 else "BACKWARD"

    if adx >= ady and adx >= dx_thr:
        return "LEFT" if dx > 0 else "RIGHT"

    if ady >= dy_thr:
        return "UP" if dy > 0 else "DOWN"

    return "STILL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Movimiento de camara con Lucas-Kanade")
    add_stream_args(parser)
    parser.add_argument("--calib", default=str(ROOT / "01_Calibracion" / "calib.txt"), help="calibracion opcional para fx/fy")
    parser.add_argument("--hfov", type=float, default=72.5, help="FOV horizontal si no hay calibracion")
    parser.add_argument("--dx-thr", type=float, default=1.4, help="umbral horizontal medio en px/frame")
    parser.add_argument("--dy-thr", type=float, default=1.4, help="umbral vertical medio en px/frame")
    parser.add_argument("--dr-thr", type=float, default=1.2, help="umbral radial medio en px/frame")
    parser.add_argument("--smooth", type=int, default=8, help="ventana de suavizado temporal")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        fx_calib, fy_calib = load_calibration(args.calib)
    except Exception:
        fx_calib, fy_calib = None, None

    tracks: list[deque[np.ndarray]] = []
    prevgray = None
    prev_t = None
    motion_hist: deque[tuple[float, float, float]] = deque(maxlen=max(1, int(args.smooth)))
    fps_est = 25.0

    for n, (key, frame) in enumerate(get_stream(args)):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        cx, cy = 0.5 * w, 0.5 * h
        center_xy = np.array([cx, cy], dtype=np.float32)

        now = time.time()
        if prev_t is None:
            prev_t = now
        dt = max(1e-3, now - prev_t)
        prev_t = now
        fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

        if prevgray is None or key == ord("c"):
            tracks.clear()
            prevgray = gray
            motion_hist.clear()

        p0_good = np.empty((0, 2), dtype=np.float32)
        p1_good = np.empty((0, 2), dtype=np.float32)
        mean_flow = np.zeros(2, dtype=np.float32)
        radial_change = 0.0
        track_ms = 0.0

        t0 = time.time()
        if tracks:
            p0 = np.float32([t[-1] for t in tracks])
            p1, _, _ = cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **LK_PARAMS)
            p0r, _, _ = cv.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **LK_PARAMS)
            d = abs(p0 - p0r).reshape(-1, 2).max(axis=1)
            good = d < 1

            new_tracks: list[deque[np.ndarray]] = []
            p0_list = []
            p1_list = []
            for t, point0, point1, ok in zip(tracks, p0.reshape(-1, 2), p1.reshape(-1, 2), good):
                if not ok:
                    continue
                t.append(point1)
                new_tracks.append(t)
                p0_list.append(point0)
                p1_list.append(point1)

            tracks = new_tracks
            if p0_list:
                p0_good = np.asarray(p0_list, dtype=np.float32)
                p1_good = np.asarray(p1_list, dtype=np.float32)
                mean_flow, _kept = robust_center_flow(p0_good, p1_good)
                radial_change = robust_radial_change(p0_good, p1_good, center_xy)

            cv.polylines(frame, [np.int32(t) for t in tracks], isClosed=False, color=(0, 0, 255))
            for t in tracks:
                point = np.int32(t[-1])
                cv.circle(frame, center=tuple(point), radius=2, color=(0, 0, 255), thickness=-1)

        track_ms = (time.time() - t0) * 1000.0

        if n % DETECT_INTERVAL == 0:
            mask = np.zeros_like(gray)
            mask[:] = 255
            for x, y in [np.int32(t[-1]) for t in tracks]:
                cv.circle(mask, (x, y), 5, 0, -1)
            corners = cv.goodFeaturesToTrack(gray, mask=mask, **CORNERS_PARAMS)
            if corners is not None:
                for [pt] in np.float32(corners):
                    tracks.append(deque([pt], maxlen=TRACK_LEN))

        motion_hist.append((float(mean_flow[0]), float(mean_flow[1]), float(radial_change)))
        sx = float(np.mean([m[0] for m in motion_hist])) if motion_hist else 0.0
        sy = float(np.mean([m[1] for m in motion_hist])) if motion_hist else 0.0
        sr = float(np.mean([m[2] for m in motion_hist])) if motion_hist else 0.0

        motion_label = classify_motion(
            dx=sx,
            dy=sy,
            dr=sr,
            dx_thr=float(args.dx_thr),
            dy_thr=float(args.dy_thr),
            dr_thr=float(args.dr_thr),
        )

        if fx_calib is None or fy_calib is None:
            fx = w / (2.0 * math.tan(math.radians(float(args.hfov)) / 2.0))
            fy = fx
        else:
            fx = fx_calib
            fy = fy_calib

        yaw_rate, pitch_rate, total_rate = estimate_deg_per_sec(sx, sy, dt, fx, fy)

        arrow_start = (int(round(cx)), int(round(cy)))
        arrow_end = (int(round(cx + sx * 20.0)), int(round(cy + sy * 20.0)))
        cv.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 2, cv.LINE_AA, tipLength=0.25)
        cv.circle(frame, arrow_start, 4, (0, 255, 255), -1, cv.LINE_AA)

        cv.putText(frame, f"tracks={len(tracks)}  {track_ms:.0f}ms  fps={fps_est:.1f}", (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, f"CAMERA: {motion_label}", (10, 52), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv.LINE_AA)
        cv.putText(frame, f"flow mean: dx={sx:+.2f}px  dy={sy:+.2f}px  dr={sr:+.2f}px", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv.LINE_AA)
        cv.putText(frame, f"yaw={yaw_rate:+.2f} deg/s  pitch={pitch_rate:+.2f} deg/s  ang={total_rate:.2f} deg/s", (10, 106), cv.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv.LINE_AA)
        cv.putText(frame, "C reinicia tracks | Q o ESC sale", (10, h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv.LINE_AA)

        cv.imshow("LK Camera Motion", frame)
        prevgray = gray

        if key in (27, ord("q")):
            break

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
