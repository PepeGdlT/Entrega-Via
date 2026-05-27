#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


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
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir --dev={args.dev}")

        def fallback():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                key = cv2.waitKey(1) & 0xFF
                yield key, frame
            cap.release()

        return fallback()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prueba en vivo del detector YOLO de taza")
    add_stream_args(parser)
    parser.add_argument("--model", default=str(Path(__file__).with_name("models") / "taza.pt"), help="modelo .pt")
    parser.add_argument("--conf", type=float, default=0.35, help="confianza minima")
    parser.add_argument("--imgsz", type=int, default=640, help="tamano de entrada")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"No existe el modelo: {model_path}")

    model = YOLO(str(model_path))
    fps_est = 25.0
    t_prev = time.time()

    for key, frame in get_stream(args):
        t_now = time.time()
        dt = max(1e-3, t_now - t_prev)
        t_prev = t_now
        fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, conf=float(args.conf), imgsz=int(args.imgsz), verbose=False)
        plotted = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
        n = 0 if results[0].boxes is None else len(results[0].boxes)

        cv2.putText(
            plotted,
            f"model={model_path.name} conf={args.conf:.2f} detections={n} fps={fps_est:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            plotted,
            "Q o ESC para salir",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
        )
        cv2.imshow("YOLO taza", plotted)

        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
