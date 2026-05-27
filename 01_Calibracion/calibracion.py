#!/usr/bin/env python
"""Ejercicio 01 - Calibracion y rejilla metrica interactiva.

Modo calibrate:
- Detecta un chessboard en la fuente de entrada.
- Permite capturar poses con la tecla 'c'.
- Calcula K y distorsion con 'ENTER' y guarda en un txt compatible con umucv.

Modo overlay:
- Carga la calibracion.
- Dibuja una rejilla metrica sobre un plano a distancia Z.
- Muestra FOV horizontal/vertical y ayudas de alineacion.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

# Permite ejecutar sin instalar umucv globalmente.
ROOT = Path(__file__).resolve().parents[1]
UMUCV_PKG = ROOT / "umucv" / "package"
if str(UMUCV_PKG) not in sys.path:
    sys.path.append(str(UMUCV_PKG))

from umucv.stream import autoStream, sourceArgs  # type: ignore
from umucv.util import Slider, putText  # type: ignore


def parse_pattern(pattern_text: str) -> tuple[int, int]:
    for sep in ("x", "X", ","):
        if sep in pattern_text:
            a, b = pattern_text.split(sep, 1)
            cols, rows = int(a), int(b)
            if cols < 2 or rows < 2:
                raise argparse.ArgumentTypeError("pattern debe ser >= 2x2")
            return cols, rows
    raise argparse.ArgumentTypeError("pattern debe tener formato 9x6")


def build_pattern_points(pattern_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2)
    pts *= square_size
    return pts


def compute_fov_deg(k: np.ndarray, width: int, height: int) -> tuple[float, float]:
    fx, fy = float(k[0, 0]), float(k[1, 1])
    hfov = 2.0 * math.degrees(math.atan(width / (2.0 * fx)))
    vfov = 2.0 * math.degrees(math.atan(height / (2.0 * fy)))
    return hfov, vfov


def save_calibration(path: Path, k: np.ndarray, d: np.ndarray) -> None:
    data = np.concatenate([k.flatten(), d.flatten()])
    np.savetxt(path, data)


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(path).astype(np.float64).flatten()
    if raw.size < 14:
        raise ValueError(f"Calibracion invalida en {path}: minimo 14 valores")
    k = raw[:9].reshape(3, 3)
    d = raw[9:]
    return k, d


def calibrate_mode(args: argparse.Namespace) -> int:
    pattern_pts = build_pattern_points(args.pattern, args.square_size)
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    print("[calibrate] Pulsa 'c' para capturar una pose valida del chessboard.")
    print("[calibrate] Pulsa ENTER para calibrar y guardar. ESC/q para salir.")

    for key, frame in autoStream():
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h, w = gray.shape
        image_size = (w, h)

        found, corners = cv.findChessboardCorners(gray, args.pattern)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            cv.drawChessboardCorners(frame, args.pattern, corners, found)

        if key == ord("c"):
            if found:
                img_points.append(corners.reshape(-1, 2).astype(np.float32))
                obj_points.append(pattern_pts.copy())
                print(f"[calibrate] Captura {len(obj_points)} guardada")
            else:
                print("[calibrate] No se detecta tablero en este frame")

        putText(frame, f"capturas: {len(obj_points)} (min {args.min_views})", (8, 18))
        putText(frame, "c:capturar ENTER:calibrar q/ESC:salir", (8, 38), scale=0.9)
        cv.imshow("calibracion", frame)

        if key in (13, 10):
            if len(obj_points) < args.min_views:
                print(f"[calibrate] Faltan capturas: {len(obj_points)}/{args.min_views}")
                continue
            if image_size is None:
                print("[calibrate] No hay tamano de imagen valido")
                return 1

            rms, k, d, _, _ = cv.calibrateCamera(obj_points, img_points, image_size, None, None)
            hfov, vfov = compute_fov_deg(k, image_size[0], image_size[1])

            print(f"[calibrate] RMS: {rms:.4f}")
            print("[calibrate] K:")
            print(np.array2string(k, formatter={"float_kind": lambda x: f"{x:9.3f}"}))
            print(f"[calibrate] D: {np.round(d.flatten(), 6)}")
            print(f"[calibrate] FOV: h={hfov:.2f} deg, v={vfov:.2f} deg")

            save_calibration(Path(args.calib), k, d)
            print(f"[calibrate] Guardado en: {args.calib}")
            return 0

    return 0


def project_points(
    points_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_height: float,
    pitch_deg: float,
) -> np.ndarray:
    """Proyecta puntos de un plano z=const en el sistema camara.

    Sistema mundo:
    - X: derecha (m)
    - Y: arriba (m)
    - Z: frente (m)

    Sistema camara OpenCV:
    - x: derecha
    - y: abajo
    - z: frente
    """
    xw = points_world[:, 0]
    yw = points_world[:, 1]
    zw = points_world[:, 2]

    # Traslacion mundo->camara (camara a altura cam_height sobre el suelo).
    xc = xw
    yc = cam_height - yw
    zc = zw

    # Correccion de inclinacion de camara alrededor del eje X.
    a = math.radians(pitch_deg)
    ca, sa = math.cos(a), math.sin(a)
    yr = ca * yc - sa * zc
    zr = sa * yc + ca * zc

    zr = np.maximum(zr, 1e-6)
    u = fx * (xc / zr) + cx
    v = fy * (yr / zr) + cy
    return np.vstack((u, v)).T


def draw_metric_grid(
    frame: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    z_plane: float,
    cam_height: float,
    pitch_deg: float,
    x_offset: float,
    step: float,
    x_extent: float,
    y_max: float,
) -> None:
    # Ejes de rejilla: x horizontal, y vertical (altura sobre suelo).
    x_values = np.arange(-x_extent, x_extent + 1e-9, step) + x_offset
    y_values = np.arange(0.0, y_max + 1e-9, step)

    # Lineas verticales de rejilla.
    for x in x_values:
        p0 = np.array([[x, 0.0, z_plane], [x, y_max, z_plane]], dtype=np.float32)
        q = project_points(p0, fx, fy, cx, cy, cam_height, pitch_deg).astype(int)
        cv.line(frame, tuple(q[0]), tuple(q[1]), (220, 220, 220), 1, cv.LINE_AA)

    # Lineas horizontales de rejilla.
    for y in y_values:
        p0 = np.array([[-x_extent + x_offset, y, z_plane], [x_extent + x_offset, y, z_plane]], dtype=np.float32)
        q = project_points(p0, fx, fy, cx, cy, cam_height, pitch_deg).astype(int)
        thick = 3 if abs(y) < 1e-6 else 1
        color = (255, 255, 255) if thick == 3 else (210, 210, 210)
        cv.line(frame, tuple(q[0]), tuple(q[1]), color, thick, cv.LINE_AA)

    # Etiquetas de altura cada 1 metro.
    y_label = np.arange(0.0, y_max + 1e-9, 1.0)
    for y in y_label:
        p = np.array([[0.0 + x_offset, y, z_plane]], dtype=np.float32)
        q = project_points(p, fx, fy, cx, cy, cam_height, pitch_deg)[0].astype(int)
        cv.putText(frame, f"{int(y)}", tuple(q + np.array([5, -5])), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1, cv.LINE_AA)


def overlay_mode(args: argparse.Namespace) -> int:
    k, _d = load_calibration(Path(args.calib))

    # Ventana principal y sliders estilo umucv.
    wnd = "medidor"
    fov0, _, = compute_fov_deg(k, 640, 480)  # inicial; se ajusta al primer frame real.
    fov_min, fov_max = 20.0, 120.0
    hfov_deg = float(np.clip(round(fov0), fov_min, fov_max))

    # FOV en grados reales para que la trackbar no muestre indices escalados.
    cv.namedWindow(wnd)
    cv.createTrackbar("fov", wnd, int(hfov_deg), int(fov_max), lambda _v: None)
    cv.setTrackbarMin("fov", wnd, int(fov_min))
    s_z = Slider("Z", wnd, args.z0, 0.5, 30.0, 0.1)
    s_h = Slider("A", wnd, args.height0, 0.0, 3.0, 0.01)
    s_x = Slider("X", wnd, args.x0, -5.0, 5.0, 0.05)

    first_frame = True
    ratio_fy_fx = float(k[1, 1] / k[0, 0])

    for key, frame in autoStream():
        h, w = frame.shape[:2]

        if first_frame:
            fov_real_h, fov_real_v = compute_fov_deg(k, w, h)
            # Inicializa FOV del slider con el valor calibrado real de esta resolucion.
            if fov_min <= fov_real_h <= fov_max:
                hfov_deg = float(round(fov_real_h))
                cv.setTrackbarPos("fov", wnd, int(hfov_deg))
            print(f"[overlay] Resolucion: {w}x{h}")
            print(f"[overlay] FOV calibrado: h={fov_real_h:.2f} deg, v={fov_real_v:.2f} deg")
            first_frame = False

        hfov = float(cv.getTrackbarPos("fov", wnd))
        z_plane = float(s_z.value)
        cam_height = float(s_h.value)
        x_offset = float(s_x.value)

        fx = w / (2.0 * math.tan(math.radians(hfov) / 2.0))
        fy = fx * ratio_fy_fx
        cx = float(k[0, 2])
        cy = float(k[1, 2])

        pitch_deg = float(args.pitch)

        draw_metric_grid(
            frame=frame,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            z_plane=z_plane,
            cam_height=cam_height,
            pitch_deg=pitch_deg,
            x_offset=x_offset,
            step=args.grid_step,
            x_extent=args.grid_half_width,
            y_max=args.grid_height,
        )

        # Referencia del centro de camara (debe coincidir con linea de altura A cuando pitch~0).
        cv.line(frame, (0, int(round(cy))), (w - 1, int(round(cy))), (160, 160, 160), 1, cv.LINE_AA)

        # Texto informativo.
        vfov = 2.0 * math.degrees(math.atan(h / (2.0 * fy)))
        putText(frame, f"FOV={hfov:.1f} deg, f={fx:.0f}px ({w}x{h})", (8, 18))
        putText(frame, f"Z={z_plane:.1f} m", (8, 40))
        putText(frame, f"alt={cam_height:.2f} m", (8, 62))
        putText(frame, f"vfov={vfov:.1f} deg", (8, 84))

        cv.imshow(wnd, frame)

        if key in (ord("q"), 27):
            break

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibracion + rejilla metrica para VIA")
    sourceArgs(parser)

    parser.add_argument("--mode", choices=("calibrate", "overlay"), default="overlay")
    parser.add_argument("--calib", default=str(Path(__file__).with_name("calib.txt")), help="archivo de calibracion txt")

    # Parametros de calibracion chessboard.
    parser.add_argument("--pattern", type=parse_pattern, default="9x6", help="tamano interior tablero, ej. 9x6")
    parser.add_argument("--square-size", type=float, default=1.0, help="tamano real de celda (unidad libre, p.ej. metros)")
    parser.add_argument("--min-views", type=int, default=12, help="capturas minimas para calibrar")

    # Parametros del overlay.
    parser.add_argument("--z0", type=float, default=2.0, help="distancia inicial del plano Z")
    parser.add_argument("--height0", type=float, default=0.8, help="altura inicial de camara en metros")
    parser.add_argument("--x0", type=float, default=0.0, help="desplazamiento lateral inicial (m)")
    parser.add_argument("--pitch", type=float, default=0.0, help="correccion fija de inclinacion (grados)")
    parser.add_argument("--grid-step", type=float, default=0.5, help="paso de rejilla en metros")
    parser.add_argument("--grid-half-width", type=float, default=3.0, help="semiancho de rejilla en metros")
    parser.add_argument("--grid-height", type=float, default=3.0, help="altura de rejilla en metros")

    args, rest = parser.parse_known_args()
    if len(rest) > 0:
        raise SystemExit("unknown parameters: " + str(rest))

    return args


def main() -> int:
    args = parse_args()
    if args.mode == "calibrate":
        return calibrate_mode(args)
    return overlay_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())

