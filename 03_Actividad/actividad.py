import argparse
import itertools
import os
import sys
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests


WINDOW_NAME = "Actividad - TownCentre"
EVENT_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "eventos")
MODEL_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "models")
YOLO_MODEL_DEFAULT = os.path.join(MODEL_DIR_DEFAULT, "yolov8n.pt")

# Configuracion base (sin necesidad de argumentos CLI).
DEFAULT_PIPELINE = {
    "yolo_model": YOLO_MODEL_DEFAULT,
    "device": "cpu",
    "input_size": 640,
    "mog2_history": 500,
    "mog2_var_threshold": 25.0,
    "event_seconds": 3.0,
    "prebuffer_seconds": 1.0,
    "cooldown_seconds": 5.0,
    "output_dir": EVENT_DIR_DEFAULT,
}

# Valores iniciales de barras (modo simple: solo lo importante en vivo).
DEFAULT_RUNTIME = {
    "conf": 0.28,
    "dnn_every": 2,
    "motion_area_min": 900,
    "red_ratio_thr": 0.25,
}

# Parametros avanzados fijos para no saturar la interfaz.
DEFAULT_FIXED = {
    "motion_thr": 170,
    "red_center_ratio_thr": 0.02,
    "red_motion_ratio_thr": 0.005,
    "red_s_min": 55,
    "red_v_min": 50,
    "red_center_scale": 0.65,
    "anon_hold_frames": 12,
    "track_iou_thr": 0.32,
}


def _ensure_umucv_package():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    package_dir = os.path.join(repo_root, "umucv", "package")
    if os.path.isdir(package_dir) and package_dir not in sys.path:
        sys.path.insert(0, package_dir)


def _add_stream_args(parser):
    try:
        _ensure_umucv_package()
        from umucv.stream import sourceArgs

        sourceArgs(parser)
    except Exception:
        parser.add_argument("--dev", type=str, default="0", help="fuente de imagen")
        parser.add_argument("--size", default=None)
        parser.add_argument("--resize", default=None)
        parser.add_argument("--step", action="store_true")
        parser.add_argument("--loop", action="store_true")


def _get_stream(args):
    try:
        _ensure_umucv_package()
        from umucv.stream import autoStream

        return autoStream()
    except Exception:
        src = args.dev
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir --dev={args.dev}")

        def fallback_stream():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                key = cv2.waitKey(1) & 0xFF
                yield key, frame
            cap.release()

        return fallback_stream()


def _make_roi_controller(window_name):
    try:
        _ensure_umucv_package()
        from umucv.util import ROI

        cv2.namedWindow(window_name)
        return ROI(window_name), True
    except Exception:
        return None, False


class SimpleROI:
    def __init__(self, x1, y1, x2, y2):
        self.roi = [int(x1), int(y1), int(x2), int(y2)]


class ROIEditor:
    def __init__(self, window_name):
        self.dragging = False
        self.start = None
        self.current = None
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (int(x), int(y))
            self.current = (int(x), int(y))
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.current = (int(x), int(y))

    def consume_drawn_roi(self, frame_w, frame_h):
        if self.start is None or self.current is None:
            return None
        x1, y1 = self.start
        x2, y2 = self.current
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(x1, frame_w - 1))
        x2 = max(0, min(x2, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        y2 = max(0, min(y2, frame_h - 1))
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return None
        return (x1, y1, x2, y2)

    def clear(self):
        self.start = None
        self.current = None
        self.dragging = False

    def draw_preview(self, frame):
        if self.start is None or self.current is None:
            return
        x1, y1 = self.start
        x2, y2 = self.current
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)


def _read_roi(region, frame):
    if region and getattr(region, "roi", None):
        x1, y1, x2, y2 = region.roi
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        if w > 1 and h > 1:
            return x1, y1, w, h
    return None


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def create_runtime_sliders(window_name, init_cfg):
    cv2.createTrackbar("conf x100", window_name, int(_clamp(init_cfg["conf"] * 100, 1, 100)), 100, lambda _v: None)
    cv2.createTrackbar("dnn every", window_name, int(_clamp(init_cfg["dnn_every"], 1, 12)), 12, lambda _v: None)
    cv2.createTrackbar(
        "motion area min",
        window_name,
        int(_clamp(init_cfg["motion_area_min"], 0, 20000)),
        20000,
        lambda _v: None,
    )
    cv2.createTrackbar("red ratio x100", window_name, int(_clamp(init_cfg["red_ratio_thr"] * 100, 0, 100)), 100, lambda _v: None)


def read_runtime_sliders(window_name):
    dnn_every = max(1, cv2.getTrackbarPos("dnn every", window_name))

    return {
        "conf": cv2.getTrackbarPos("conf x100", window_name) / 100.0,
        "dnn_every": dnn_every,
        "motion_area_min": cv2.getTrackbarPos("motion area min", window_name),
        "red_ratio_thr": cv2.getTrackbarPos("red ratio x100", window_name) / 100.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detector de movimiento en ROI + persona con rojo + alerta Telegram (CLI minima)"
    )
    _add_stream_args(parser)
    parser.add_argument(
        "--video",
        default=None,
        help="Compatibilidad antigua. Se recomienda usar --dev.",
    )

    parser.add_argument(
        "--telegram-token",
        default=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        help="Token de bot Telegram o variable TELEGRAM_BOT_TOKEN",
    )
    parser.add_argument(
        "--telegram-chat-id",
        default=os.environ.get("TELEGRAM_CHAT_ID", ""),
        help="Chat ID Telegram o variable TELEGRAM_CHAT_ID",
    )
    parser.add_argument("--no-telegram", action="store_true", help="No enviar por Telegram")
    parser.add_argument("--self-test", action="store_true")
    args, rest = parser.parse_known_args(sys.argv)
    unknown = [p for p in rest[1:] if p and p.strip() and p != "`"]
    if unknown:
        parser.error("unknown parameters: " + str(unknown))

    if args.video:
        args.dev = args.video
    elif getattr(args, "dev", "default") == "default":
        args.dev = os.path.join(os.path.dirname(__file__), "videos", "TownCentre.mp4")
    return args


def load_yolo(model_path):
    from ultralytics import YOLO

    return YOLO(model_path)


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def blur_box(frame, box, ksize=31):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, w, h)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    # El kernel impar evita error en GaussianBlur.
    k = max(3, int(ksize) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def intersection_ratio(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / float(area_a)


def detect_motion_in_roi(frame, roi, backsub, thr_bin, area_min):
    x, y, w, h = roi
    roi_img = frame[y : y + h, x : x + w]

    fg = backsub.apply(roi_img)
    _, fg = cv2.threshold(fg, int(thr_bin), 255, cv2.THRESH_BINARY)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k7)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        rx, ry, rw, rh = cv2.boundingRect(c)
        motion_boxes.append((x + rx, y + ry, x + rx + rw, y + ry + rh))

    return len(motion_boxes) > 0, motion_boxes, fg


def red_ratio_hsv(crop_bgr, s_min=70, v_min=60):
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, s_min, v_min], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    m1 = cv2.inRange(hsv, lower_red1, upper_red1)
    m2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(m1, m2)

    red_pixels = int(np.count_nonzero(mask))
    total_pixels = mask.size
    return red_pixels / float(max(1, total_pixels))


def red_mask_hsv(crop_bgr, s_min=70, v_min=60):
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, s_min, v_min], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    m1 = cv2.inRange(hsv, lower_red1, upper_red1)
    m2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(m1, m2)


def crop_center(box, scale):
    x1, y1, x2, y2 = box
    scale = max(0.3, min(float(scale), 1.0))
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    return (int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2))


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def update_anonymization_tracks(active_tracks, detected_people, hold_frames, iou_thr):
    hold_frames = max(1, int(hold_frames))
    iou_thr = max(0.05, min(float(iou_thr), 0.95))
    matched = set()

    for tr in active_tracks:
        best_iou = 0.0
        best_j = -1
        for j, det in enumerate(detected_people):
            if j in matched:
                continue
            score = box_iou(tr["box"], det["box"])
            if score > best_iou:
                best_iou = score
                best_j = j
        tr["ttl"] -= 1
        if best_j >= 0 and best_iou >= iou_thr:
            d = detected_people[best_j]
            tr["box"] = d["box"]
            tr["conf"] = d["conf"]
            tr["red_ratio"] = d["red_ratio"]
            tr["ttl"] = hold_frames
            matched.add(best_j)

    for j, d in enumerate(detected_people):
        if j in matched:
            continue
        active_tracks.append(
            {
                "box": d["box"],
                "conf": d["conf"],
                "red_ratio": d["red_ratio"],
                "ttl": hold_frames,
            }
        )

    return [tr for tr in active_tracks if tr["ttl"] > 0]


def detect_red_persons(
    frame,
    yolo_model,
    conf,
    device,
    imgsz,
    motion_boxes,
    motion_mask,
    red_ratio_thr,
    red_center_ratio_thr,
    red_motion_ratio_thr,
    red_center_scale,
    s_min,
    v_min,
):
    # YOLO se ejecuta solo cuando ya hay movimiento, para ahorrar tiempo.
    results = yolo_model.predict(frame, conf=conf, classes=[0], device=device, imgsz=imgsz, verbose=False)
    if not results:
        return []

    people = []
    boxes = results[0].boxes
    if boxes is None:
        return people

    h, w = frame.shape[:2]
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = clamp_box(xyxy, w, h)
        person_box = (x1, y1, x2, y2)

        # Exigimos solape con movimiento para evitar disparos espurios.
        overlaps_motion = any(intersection_ratio(person_box, mb) > 0.1 for mb in motion_boxes)
        if not overlaps_motion:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        rr = red_ratio_hsv(crop, s_min=s_min, v_min=v_min)
        cx1, cy1, cx2, cy2 = clamp_box(crop_center(person_box, red_center_scale), w, h)
        center_crop = frame[cy1:cy2, cx1:cx2]
        rr_center = red_ratio_hsv(center_crop, s_min=s_min, v_min=v_min)

        red_mask = red_mask_hsv(crop, s_min=s_min, v_min=v_min)
        rr_motion = 0.0
        if motion_mask is not None and red_mask is not None:
            mm = motion_mask[y1:y2, x1:x2]
            if mm.size > 0:
                moving = mm > 0
                if np.count_nonzero(moving) > 0:
                    rr_motion = float(np.count_nonzero((red_mask > 0) & moving)) / float(red_mask.size)

        if rr >= red_ratio_thr and rr_center >= red_center_ratio_thr and rr_motion >= red_motion_ratio_thr:
            people.append(
                {
                    "box": person_box,
                    "conf": float(box.conf[0]),
                    "red_ratio": rr,
                }
            )

    return people


def save_event_video(frames, out_path, fps):
    if not frames:
        return False

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(8.0, float(fps)),
        (w, h),
    )
    for fr in frames:
        writer.write(fr)
    writer.release()
    return True


def send_telegram_photo(token, chat_id, image_path, caption):
    if not token or not chat_id:
        return False, "Telegram no configurado"
    try:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": chat_id, "caption": caption}
            response = requests.post(url, data=data, files=files, timeout=10)
        if response.status_code == 200:
            return True, "Telegram enviado"
        return False, f"Telegram HTTP {response.status_code}: {response.text[:120]}"
    except Exception as exc:
        return False, f"Error Telegram: {exc}"


def draw_ui(frame, roi, motion_boxes, red_people, state_text):
    x, y, w, h = roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    for mb in motion_boxes:
        x1, y1, x2, y2 = mb
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 0), 2)

    for p in red_people:
        x1, y1, x2, y2 = p["box"]
        txt = f"person {p['conf']:.2f} red={p['red_ratio']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, txt, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    cv2.putText(frame, state_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def run_self_test():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), -1)
    ratio = red_ratio_hsv(img)
    assert ratio > 0.2

    textured = np.zeros((80, 80, 3), dtype=np.uint8)
    patch = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)
    textured[20:60, 20:60] = patch
    before = textured.copy()
    blur_box(textured, (20, 20, 60, 60))
    assert not np.array_equal(before, textured)

    assert intersection_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert intersection_ratio((0, 0, 20, 20), (10, 10, 25, 25)) > 0.0

    print("Self-test OK")


def main():
    args = parse_args()
    if args.self_test:
        run_self_test()
        return

    output_dir = DEFAULT_PIPELINE["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(MODEL_DIR_DEFAULT, exist_ok=True)

    stream = iter(_get_stream(args))

    print("Selecciona ROI en el primer frame. ENTER/SPACE para empezar. R para editar ROI en ejecucion.")
    print("Barras activas: conf, dnn every, motion area min, red ratio.")
    cv2.namedWindow(WINDOW_NAME)
    region, roi_mode_umucv = _make_roi_controller(WINDOW_NAME)
    roi_editor = ROIEditor(WINDOW_NAME)
    create_runtime_sliders(WINDOW_NAME, DEFAULT_RUNTIME)

    try:
        first_key, first_frame = next(stream)
    except StopIteration:
        print("No hay frames en la fuente de video")
        return

    while True:
        boot = first_frame.copy()
        roi_editor.draw_preview(boot)
        roi = _read_roi(region, boot) if roi_mode_umucv else None
        if roi:
            x, y, w, h = roi
            cv2.rectangle(boot, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            boot,
            "Arrastra ROI y ENTER/SPACE para iniciar | X limpia | Q/ESC salir",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imshow(WINDOW_NAME, boot)

        h0, w0 = first_frame.shape[:2]
        drawn = roi_editor.consume_drawn_roi(w0, h0)
        if drawn is not None:
            region = SimpleROI(*drawn)
            roi_mode_umucv = True

        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32) and _read_roi(region, first_frame):
            break
        if key == ord("x"):
            roi_editor.clear()
            if region is not None:
                region.roi = []
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            return

    stream = itertools.chain([(first_key, first_frame)], stream)

    backsub = cv2.createBackgroundSubtractorMOG2(
        history=int(DEFAULT_PIPELINE["mog2_history"]),
        varThreshold=float(DEFAULT_PIPELINE["mog2_var_threshold"]),
        detectShadows=True,
    )

    yolo_model = load_yolo(DEFAULT_PIPELINE["yolo_model"])

    fps = 25.0

    prebuffer = deque()
    dnn_every = max(1, int(DEFAULT_RUNTIME["dnn_every"]))
    frame_idx = 0
    last_t = time.time()
    fps_est = fps

    recording = False
    event_frames = []
    record_until = 0.0
    cooldown_until = 0.0
    event_id = ""
    active_tracks = []
    roi_edit_mode = False

    for key, frame in stream:

        runtime = {**DEFAULT_FIXED, **read_runtime_sliders(WINDOW_NAME)}
        dnn_every = int(runtime["dnn_every"])

        frame_idx += 1
        now = time.time()
        dt = max(1e-3, now - last_t)
        last_t = now
        fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

        if key == ord("r"):
            roi_edit_mode = not roi_edit_mode

        h_frame, w_frame = frame.shape[:2]
        drawn = roi_editor.consume_drawn_roi(w_frame, h_frame)
        if drawn is not None:
            region = SimpleROI(*drawn)
            roi_mode_umucv = True
            backsub = cv2.createBackgroundSubtractorMOG2(
                history=int(DEFAULT_PIPELINE["mog2_history"]),
                varThreshold=float(DEFAULT_PIPELINE["mog2_var_threshold"]),
                detectShadows=True,
            )

        roi = _read_roi(region, frame) if roi_mode_umucv else None
        if not roi or roi_edit_mode:
            vis_wait = frame.copy()
            if roi:
                x, y, w, h = roi
                cv2.rectangle(vis_wait, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_editor.draw_preview(vis_wait)
            msg = "EDIT ROI ON | Arrastra ROI | R salir | X limpia | Q salir" if roi_edit_mode else "Dibuja ROI para continuar"
            cv2.putText(vis_wait, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(WINDOW_NAME, vis_wait)
            cv2.imshow("Mascara ROI", np.zeros(frame.shape[:2], dtype=np.uint8))
            if key == ord("x") and region is not None:
                region.roi = []
                active_tracks = []
            if key == 27 or key == ord("q"):
                break
            continue

        x, y, w, h = roi
        motion_found, motion_boxes, fg = detect_motion_in_roi(
            frame,
            roi,
            backsub,
            thr_bin=runtime["motion_thr"],
            area_min=runtime["motion_area_min"],
        )

        motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        motion_mask[y : y + h, x : x + w] = fg

        red_people = []
        if motion_found and (frame_idx % dnn_every == 0):
            red_people = detect_red_persons(
                frame,
                yolo_model,
                conf=float(runtime["conf"]),
                device=DEFAULT_PIPELINE["device"],
                imgsz=int(DEFAULT_PIPELINE["input_size"]),
                motion_boxes=motion_boxes,
                motion_mask=motion_mask,
                red_ratio_thr=float(runtime["red_ratio_thr"]),
                red_center_ratio_thr=float(runtime["red_center_ratio_thr"]),
                red_motion_ratio_thr=float(runtime["red_motion_ratio_thr"]),
                red_center_scale=float(runtime["red_center_scale"]),
                s_min=int(runtime["red_s_min"]),
                v_min=int(runtime["red_v_min"]),
            )

        active_tracks = update_anonymization_tracks(
            active_tracks,
            red_people,
            hold_frames=int(runtime["anon_hold_frames"]),
            iou_thr=float(runtime["track_iou_thr"]),
        )

        vis = frame.copy()
        evidence = frame.copy()

        for tr in active_tracks:
            blur_box(evidence, tr["box"])
            blur_box(vis, tr["box"])

        prebuffer.append((now, evidence.copy()))
        while prebuffer and (now - prebuffer[0][0] > float(DEFAULT_PIPELINE["prebuffer_seconds"])):
            prebuffer.popleft()

        can_trigger = now >= cooldown_until
        trigger = motion_found and len(active_tracks) > 0 and can_trigger and (not recording)

        if trigger:
            event_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_until = now + float(DEFAULT_PIPELINE["event_seconds"])
            recording = True
            event_frames = [fr.copy() for _, fr in prebuffer]

            img_path = os.path.join(output_dir, f"evento_{event_id}.jpg")
            cv2.imwrite(img_path, evidence)

            if args.no_telegram:
                print("Telegram desactivado (--no-telegram)")
            else:
                ok_t, msg_t = send_telegram_photo(
                    args.telegram_token,
                    args.telegram_chat_id,
                    img_path,
                    caption=f"Evento detectado ({event_id}) persona con rojo",
                )
                print(msg_t)
                if not ok_t:
                    print(f"Imagen guardada localmente: {img_path}")

        if recording:
            event_frames.append(evidence.copy())
            if motion_found and active_tracks:
                record_until = max(record_until, now + 0.4)

            if now >= record_until:
                video_path = os.path.join(output_dir, f"evento_{event_id}.mp4")
                if save_event_video(event_frames, video_path, fps_est):
                    print(f"Video guardado: {video_path}")
                recording = False
                cooldown_until = now + float(DEFAULT_PIPELINE["cooldown_seconds"])
                event_frames = []

        state = "IDLE"
        if recording:
            state = "RECORDING"
        elif now < cooldown_until:
            state = "COOLDOWN"

        draw_ui(vis, (x, y, w, h), motion_boxes, active_tracks, f"{state} | FPS {fps_est:.1f}")
        cv2.putText(
            vis,
            f"conf={runtime['conf']:.2f} dnn={runtime['dnn_every']} area={runtime['motion_area_min']} red={runtime['red_ratio_thr']:.2f}",
            (10, 79),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        if roi_edit_mode:
            cv2.putText(vis, "EDIT ROI ON", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        # Ventanas de depuracion.
        mask_canvas = motion_mask

        cv2.imshow(WINDOW_NAME, vis)
        cv2.imshow("Mascara ROI", mask_canvas)

        if key == ord("x") and region is not None:
            region.roi = []
            active_tracks = []
            backsub = cv2.createBackgroundSubtractorMOG2(
                history=int(DEFAULT_PIPELINE["mog2_history"]),
                varThreshold=float(DEFAULT_PIPELINE["mog2_var_threshold"]),
                detectShadows=True,
            )

        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

