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


@dataclass
class BoardPose:
    rvec: np.ndarray
    tvec: np.ndarray
    H_img_to_board: np.ndarray
    corners: np.ndarray


@dataclass
class CubeObject:
    pos_xy: np.ndarray
    target_xy: np.ndarray
    color: tuple[int, int, int]
    size: float
    height: float


class MouseState:
    def __init__(self) -> None:
        self.cursor = np.array([0.0, 0.0], dtype=np.float32)
        self.left_click: np.ndarray | None = None
        self.right_click: np.ndarray | None = None


MODE_CREATE = "CREATE"
MODE_SELECT = "SELECT"
MODE_MOVE = "MOVE"


def on_mouse(event: int, x: int, y: int, _flags: int, state: MouseState) -> None:
    state.cursor[:] = (x, y)
    if event == cv.EVENT_LBUTTONDOWN:
        state.left_click = np.array([x, y], dtype=np.float32)
    elif event == cv.EVENT_RBUTTONDOWN:
        state.right_click = np.array([x, y], dtype=np.float32)


def parse_pattern(text: str) -> tuple[int, int]:
    for sep in ("x", "X", ","):
        if sep in text:
            a, b = text.split(sep, 1)
            return int(a), int(b)
    raise argparse.ArgumentTypeError("pattern debe tener formato 9x6")


def build_pattern_points(pattern: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.indices((cols, rows)).T.reshape(-1, 2)
    pts *= float(square_size)
    return pts


def load_calibration(path: Path, frame_shape: tuple[int, int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if path.exists():
        raw = np.loadtxt(path).astype(np.float64).flatten()
        if raw.size < 14:
            raise ValueError(f"Calibracion invalida: {path}")
        return raw[:9].reshape(3, 3), raw[9:].reshape(-1, 1)

    if frame_shape is None:
        raise FileNotFoundError(path)
    h, w = frame_shape[:2]
    hfov = math.radians(60.0)
    f = (w / 2.0) / math.tan(hfov / 2.0)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    return K, np.zeros((5, 1), dtype=np.float64)


def detect_board_pose(
    frame: np.ndarray,
    pattern: tuple[int, int],
    model_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> BoardPose | None:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    found, corners = cv.findChessboardCorners(gray, pattern)
    if not found:
        return None

    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.03)
    cv.cornerSubPix(gray, corners, (7, 7), (-1, -1), term)
    image_points = corners.reshape(-1, 2).astype(np.float32)

    ok, rvec, tvec = cv.solvePnP(model_points, image_points, K, D, flags=cv.SOLVEPNP_ITERATIVE)
    if not ok:
        return None

    board_xy = model_points[:, :2].astype(np.float32)
    H_img_to_board, _ = cv.findHomography(image_points, board_xy)
    if H_img_to_board is None:
        return None

    return BoardPose(rvec=rvec, tvec=tvec, H_img_to_board=H_img_to_board, corners=image_points)


def image_to_board(point: np.ndarray, H_img_to_board: np.ndarray) -> np.ndarray | None:
    p = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    q = cv.perspectiveTransform(p, H_img_to_board).reshape(2)
    if not np.all(np.isfinite(q)):
        return None
    return q.astype(np.float32)


def nearest_object(objects: list[CubeObject], board_xy: np.ndarray, max_dist: float) -> int | None:
    if not objects:
        return None
    dists = [float(np.linalg.norm(obj.pos_xy - board_xy)) for obj in objects]
    idx = int(np.argmin(dists))
    if dists[idx] <= max_dist:
        return idx
    return None


def make_cube(board_xy: np.ndarray, idx: int, size: float, height: float) -> CubeObject:
    colors = [(30, 210, 255), (255, 120, 40), (80, 240, 120), (230, 80, 255), (255, 220, 60)]
    return CubeObject(
        pos_xy=board_xy.copy(),
        target_xy=board_xy.copy(),
        color=colors[idx % len(colors)],
        size=float(size),
        height=float(height),
    )


def update_objects(objects: list[CubeObject], dt: float, speed: float) -> None:
    alpha = 1.0 - math.exp(-max(speed, 0.01) * dt)
    for obj in objects:
        obj.pos_xy = ((1.0 - alpha) * obj.pos_xy + alpha * obj.target_xy).astype(np.float32)


def cube_points(obj: CubeObject) -> np.ndarray:
    x, y = obj.pos_xy
    s = obj.size
    h = obj.height
    x0, x1 = x - s / 2.0, x + s / 2.0
    y0, y1 = y - s / 2.0, y + s / 2.0
    return np.array(
        [
            [x0, y0, 0],
            [x1, y0, 0],
            [x1, y1, 0],
            [x0, y1, 0],
            [x0, y0, -h],
            [x1, y0, -h],
            [x1, y1, -h],
            [x0, y1, -h],
        ],
        dtype=np.float32,
    )


def project(points3d: np.ndarray, pose: BoardPose, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    pts, _ = cv.projectPoints(points3d, pose.rvec, pose.tvec, K, D)
    return pts.reshape(-1, 2)


def draw_cube(frame: np.ndarray, obj: CubeObject, pose: BoardPose, K: np.ndarray, D: np.ndarray, selected: bool) -> None:
    pts = project(cube_points(obj), pose, K, D).astype(np.int32)
    bottom = pts[[0, 1, 2, 3]]
    top = pts[[4, 5, 6, 7]]
    color = obj.color

    cv.fillConvexPoly(frame, top, tuple(int(0.55 * c + 0.45 * 255) for c in color), cv.LINE_AA)
    cv.polylines(frame, [bottom, top], True, color, 2 if not selected else 4, cv.LINE_AA)
    for a, b in [(0, 4), (1, 5), (2, 6), (3, 7)]:
        cv.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2 if not selected else 4, cv.LINE_AA)

    center = project(np.array([[obj.pos_xy[0], obj.pos_xy[1], -obj.height]], dtype=np.float32), pose, K, D)[0].astype(int)
    cv.putText(frame, "3D", tuple(center + np.array([6, -6])), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)


def draw_target(frame: np.ndarray, obj: CubeObject, pose: BoardPose, K: np.ndarray, D: np.ndarray) -> None:
    x, y = obj.target_xy
    s = obj.size * 0.55
    target = np.array(
        [[x - s, y, 0], [x + s, y, 0], [x, y - s, 0], [x, y + s, 0]],
        dtype=np.float32,
    )
    pts = project(target, pose, K, D).astype(np.int32)
    cv.line(frame, tuple(pts[0]), tuple(pts[1]), obj.color, 1, cv.LINE_AA)
    cv.line(frame, tuple(pts[2]), tuple(pts[3]), obj.color, 1, cv.LINE_AA)


def draw_axes(frame: np.ndarray, pose: BoardPose, K: np.ndarray, D: np.ndarray, scale: float) -> None:
    axes = np.array([[0, 0, 0], [scale, 0, 0], [0, scale, 0], [0, 0, -scale]], dtype=np.float32)
    pts = project(axes, pose, K, D).astype(np.int32)
    origin = tuple(pts[0])
    cv.line(frame, origin, tuple(pts[1]), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(frame, origin, tuple(pts[2]), (0, 210, 0), 2, cv.LINE_AA)
    cv.line(frame, origin, tuple(pts[3]), (255, 0, 0), 2, cv.LINE_AA)
    cv.putText(frame, "x", tuple(pts[1]), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(frame, "y", tuple(pts[2]), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 0), 1, cv.LINE_AA)
    cv.putText(frame, "z", tuple(pts[3]), cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RA 3D: mover cubos virtuales sobre un chessboard con el raton")
    add_stream_args(parser)
    parser.add_argument("--calib", default=str(ROOT / "01_Calibracion" / "calib.txt"), help="archivo calib.txt con K y distorsion")
    parser.add_argument("--pattern", type=parse_pattern, default="9x6", help="esquinas internas del chessboard, ej. 9x6")
    parser.add_argument("--square-size", type=float, default=1.0, help="tamano de celda del tablero en unidades arbitrarias")
    parser.add_argument("--cube-size", type=float, default=0.85, help="lado del cubo en unidades del tablero")
    parser.add_argument("--cube-height", type=float, default=1.25, help="altura del cubo en unidades del tablero")
    parser.add_argument("--speed", type=float, default=5.0, help="velocidad de desplazamiento hacia el punto marcado")
    parser.add_argument("--max-objects", type=int, default=4, help="numero maximo de cubos")
    parser.add_argument("--save-dir", default=str(Path(__file__).with_name("captures")), help="carpeta para capturas")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pattern_points = build_pattern_points(args.pattern, float(args.square_size))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    stream = iter(get_stream(args))
    first_key, first_frame = next(stream)
    K, D = load_calibration(Path(args.calib), first_frame.shape)
    print("Usa el chessboard de calibracion como plano real.")
    print("Modos: N crear | E seleccionar | M mover seleccionado")
    print("Controles: click izq aplica modo | click dcho borra | C limpia | S guarda | Q/ESC sale")

    window = "RA Mouse 3D"
    mouse = MouseState()
    cv.namedWindow(window)
    cv.setMouseCallback(window, on_mouse, mouse)

    objects: list[CubeObject] = []
    selected_idx: int | None = None
    mode = MODE_CREATE
    last_pose: BoardPose | None = None
    prev_t = time.time()
    fps_est = 25.0

    pending = [(first_key, first_frame)]
    while True:
        if pending:
            key, frame = pending.pop(0)
        else:
            try:
                key, frame = next(stream)
            except StopIteration:
                break

        now = time.time()
        dt = max(1e-3, now - prev_t)
        prev_t = now
        fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

        if key in (27, ord("q")):
            break
        if key == ord("n"):
            mode = MODE_CREATE
        elif key == ord("e"):
            mode = MODE_SELECT
        elif key == ord("m"):
            mode = MODE_MOVE
        if key == ord("c"):
            objects.clear()
            selected_idx = None

        pose = detect_board_pose(frame, args.pattern, pattern_points, K, D)
        if pose is not None:
            last_pose = pose

        if mouse.left_click is not None:
            click = mouse.left_click
            mouse.left_click = None
            if last_pose is not None:
                board_xy = image_to_board(click, last_pose.H_img_to_board)
                if board_xy is not None:
                    if mode == MODE_CREATE:
                        if len(objects) >= int(args.max_objects):
                            print("Maximo de cubos alcanzado. Borra alguno o pulsa C.")
                        else:
                            objects.append(make_cube(board_xy, len(objects), float(args.cube_size), float(args.cube_height)))
                            selected_idx = len(objects) - 1
                            mode = MODE_MOVE
                    elif mode == MODE_SELECT:
                        idx = nearest_object(objects, board_xy, max_dist=1.5 * float(args.square_size))
                        if idx is not None:
                            selected_idx = idx
                            mode = MODE_MOVE
                        else:
                            print("No hay ningun cubo cerca del click para seleccionar.")
                    elif mode == MODE_MOVE:
                        if selected_idx is not None and 0 <= selected_idx < len(objects):
                            objects[selected_idx].target_xy = board_xy.copy()
                        else:
                            print("No hay cubo seleccionado. Pulsa N para crear o E para seleccionar.")

        if mouse.right_click is not None:
            click = mouse.right_click
            mouse.right_click = None
            if last_pose is not None:
                board_xy = image_to_board(click, last_pose.H_img_to_board)
                if board_xy is not None:
                    idx = nearest_object(objects, board_xy, max_dist=1.5 * float(args.square_size))
                    if idx is not None:
                        objects.pop(idx)
                        selected_idx = None

        update_objects(objects, dt, float(args.speed))

        vis = frame.copy()
        if pose is not None:
            cv.drawChessboardCorners(vis, args.pattern, pose.corners.reshape(-1, 1, 2), True)
            draw_axes(vis, pose, K, D, scale=1.5 * float(args.square_size))
            for i, obj in enumerate(objects):
                draw_target(vis, obj, pose, K, D)
                draw_cube(vis, obj, pose, K, D, selected=(i == selected_idx))
            status = "BOARD LOCK"
        else:
            status = "SEARCH CHESSBOARD"

        cursor = tuple(np.round(mouse.cursor).astype(int))
        cv.circle(vis, cursor, 4, (0, 255, 255), -1, cv.LINE_AA)
        selected_txt = "-" if selected_idx is None else str(selected_idx + 1)
        cv.putText(vis, f"{status}  mode={mode}  selected={selected_txt}  objects={len(objects)}  fps={fps_est:.1f}", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(vis, "N crear | E seleccionar | M mover seleccionado", (10, vis.shape[0] - 58), cv.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv.LINE_AA)
        cv.putText(vis, "Click izq aplica el modo actual sobre el tablero", (10, vis.shape[0] - 36), cv.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv.LINE_AA)
        cv.putText(vis, "Click dcho: borrar | C limpiar | S captura | Q/ESC salir", (10, vis.shape[0] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv.LINE_AA)

        if key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"ra_mouse3d_{stamp}.png"
            cv.imwrite(str(path), vis)
            print(f"Captura guardada: {path}")

        cv.imshow(window, vis)

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
