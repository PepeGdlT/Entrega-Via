#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
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


def build_output_path(output_dir: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"{label}_{stamp}.jpg"


def save_frame(frame, output_dir: Path, label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = build_output_path(output_dir, label)
    cv2.imwrite(str(out_path), frame)
    return out_path


def compute_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_motion_level(gray: np.ndarray, prev_gray: np.ndarray | None) -> float:
    if prev_gray is None:
        return 0.0
    diff = cv2.absdiff(gray, prev_gray)
    return float(np.mean(diff))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura imagenes para entrenar el detector YOLO de taza")
    add_stream_args(parser)
    parser.add_argument("--label", default="taza", help="prefijo de los archivos guardados")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).with_name("captures")),
        help="directorio donde se guardan las imagenes capturadas",
    )
    parser.add_argument("--interval", type=float, default=1.6, help="segundos minimos entre capturas en modo automatico")
    parser.add_argument("--settle-seconds", type=float, default=1.0, help="tiempo quieto antes de autocapturar")
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=2.2,
        help="umbral medio de diferencia entre frames para considerar la escena quieta",
    )
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=70.0,
        help="nitidez minima estimada para permitir una captura automatica",
    )
    parser.add_argument("--mirror", action="store_true", help="refleja horizontalmente la vista")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    auto_mode = False
    last_save = 0.0
    saved = 0
    prev_gray = None
    stable_since = None

    print("Controles: S guarda una imagen, A activa/desactiva autocaptura, Q o ESC sale.")
    print("Autocaptura: espera a que la escena este quieta y suficientemente enfocada antes de guardar.")

    for key, frame in get_stream(args):
        if args.mirror:
            frame = cv2.flip(frame, 1)

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = compute_sharpness(gray)
        motion_level = compute_motion_level(gray, prev_gray)
        prev_gray = gray

        is_stable = motion_level <= float(args.motion_threshold)
        if is_stable:
            if stable_since is None:
                stable_since = now
        else:
            stable_since = None

        stable_time = 0.0 if stable_since is None else (now - stable_since)
        is_sharp = sharpness >= float(args.min_sharpness)

        if key == ord("a"):
            auto_mode = not auto_mode
            last_save = 0.0
            stable_since = None
        if key == ord("s"):
            out_path = save_frame(frame, output_dir, args.label)
            saved += 1
            last_save = now
            print(f"Guardada: {out_path}")

        can_auto_save = (
            auto_mode
            and (now - last_save) >= float(args.interval)
            and stable_time >= float(args.settle_seconds)
            and is_sharp
        )
        if can_auto_save:
            out_path = save_frame(frame, output_dir, args.label)
            saved += 1
            last_save = now
            stable_since = None
            print(f"Guardada: {out_path}")

        hud = frame.copy()
        if not auto_mode:
            status = "MANUAL"
            color = (220, 220, 220)
        elif not is_stable:
            status = "QUIETO"
            color = (0, 180, 255)
        elif not is_sharp:
            status = "ESPERA ENFOQUE"
            color = (0, 180, 255)
        elif stable_time < float(args.settle_seconds):
            status = f"QUIETO {stable_time:.1f}/{args.settle_seconds:.1f}s"
            color = (0, 220, 255)
        else:
            status = "LISTO"
            color = (0, 220, 0)

        cv2.putText(
            hud,
            f"S=save A=auto[{('ON' if auto_mode else 'OFF')}] saved={saved}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            hud,
            f"status: {status}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )
        cv2.putText(
            hud,
            f"motion={motion_level:.2f}/{args.motion_threshold:.2f} sharp={sharpness:.0f}/{args.min_sharpness:.0f}",
            (10, 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
        )
        cv2.putText(
            hud,
            f"output: {output_dir}",
            (10, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )
        cv2.imshow("Captura taza", hud)

        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    print(f"Captura finalizada. Imagenes guardadas en: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
