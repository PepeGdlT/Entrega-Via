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

BOARD_SIZE = 450
CELL = BOARD_SIZE // 9
CANONICAL_QUAD = np.array(
    [[0.0, 0.0], [BOARD_SIZE - 1.0, 0.0], [BOARD_SIZE - 1.0, BOARD_SIZE - 1.0], [0.0, BOARD_SIZE - 1.0]],
    dtype=np.float32,
)


def add_stream_args(parser: argparse.ArgumentParser) -> None:
    try:
        from umucv.stream import sourceArgs  # type: ignore

        sourceArgs(parser)
    except Exception:
        parser.add_argument("--dev", default="0", help="fuente de video o imagen")


def get_stream(args: argparse.Namespace):
    try:
        from umucv.stream import autoStream  # type: ignore

        return autoStream()
    except Exception:
        src = args.dev
        if isinstance(src, str):
            path = Path(src)
            if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                frame = cv.imread(str(path))
                if frame is None:
                    raise RuntimeError(f"No se pudo abrir la imagen --dev={args.dev}")

                def image_stream():
                    while True:
                        key = cv.waitKey(30) & 0xFF
                        yield key, frame.copy()

                return image_stream()
            if src.isdigit():
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


def smooth_quad(prev: np.ndarray | None, curr: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return curr.astype(np.float32)
    return ((1.0 - alpha) * prev + alpha * curr).astype(np.float32)


def line_score(rectified_gray: np.ndarray) -> float:
    bw = cv.adaptiveThreshold(rectified_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 7)
    scores = []
    band = 3
    for i in range(10):
        p = int(round(i * BOARD_SIZE / 9))
        y0, y1 = max(0, p - band), min(BOARD_SIZE, p + band + 1)
        x0, x1 = max(0, p - band), min(BOARD_SIZE, p + band + 1)
        scores.append(float(np.mean(bw[y0:y1, :] > 0)))
        scores.append(float(np.mean(bw[:, x0:x1] > 0)))
    return float(np.mean(scores))


def line_from_segment(seg: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = seg.astype(np.float64)
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    n = math.hypot(a, b)
    if n <= 1e-9:
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)
    return np.array([a / n, b / n, c / n], dtype=np.float64)


def intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    x = np.cross(l1, l2)
    if abs(x[2]) <= 1e-9:
        return None
    return (x[:2] / x[2]).astype(np.float32)


def detect_sudoku_quad_hough(gray: np.ndarray) -> np.ndarray | None:
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blur, 45, 140)
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=55, minLineLength=70, maxLineGap=12)
    if lines is None or len(lines) < 12:
        return None

    segs = lines.reshape(-1, 4).astype(np.float32)
    lengths = np.linalg.norm(segs[:, 2:4] - segs[:, 0:2], axis=1)
    keep = lengths > np.percentile(lengths, 45)
    segs = segs[keep]
    lengths = lengths[keep]
    if len(segs) < 10:
        return None

    angles = np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0])
    angle_features = np.column_stack([np.cos(2 * angles), np.sin(2 * angles)]).astype(np.float32)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    _, labels, centers = cv.kmeans(angle_features, 2, None, criteria, 5, cv.KMEANS_PP_CENTERS)
    labels = labels.reshape(-1)

    families = []
    for lab in (0, 1):
        idx = np.where(labels == lab)[0]
        if len(idx) < 4:
            return None
        family_segs = segs[idx]
        family_lengths = lengths[idx]
        family_angles = angles[idx]
        vx = float(np.average(np.cos(family_angles), weights=family_lengths))
        vy = float(np.average(np.sin(family_angles), weights=family_lengths))
        normal = np.array([-vy, vx], dtype=np.float32)
        norm = np.linalg.norm(normal)
        if norm <= 1e-6:
            return None
        normal /= norm
        mids = 0.5 * (family_segs[:, 0:2] + family_segs[:, 2:4])
        proj = mids @ normal

        low = int(np.argmin(proj))
        high = int(np.argmax(proj))
        families.append((line_from_segment(family_segs[low]), line_from_segment(family_segs[high])))

    corners = [
        intersect_lines(families[0][0], families[1][0]),
        intersect_lines(families[0][0], families[1][1]),
        intersect_lines(families[0][1], families[1][1]),
        intersect_lines(families[0][1], families[1][0]),
    ]
    if any(p is None for p in corners):
        return None
    quad = order_quad(np.array(corners, dtype=np.float32))
    area = abs(cv.contourArea(quad))
    if area < 0.04 * gray.shape[0] * gray.shape[1]:
        return None
    if np.any(quad[:, 0] < -0.25 * gray.shape[1]) or np.any(quad[:, 0] > 1.25 * gray.shape[1]):
        return None
    if np.any(quad[:, 1] < -0.25 * gray.shape[0]) or np.any(quad[:, 1] > 1.25 * gray.shape[0]):
        return None

    H = cv.getPerspectiveTransform(quad, CANONICAL_QUAD)
    rect = cv.warpPerspective(gray, H, (BOARD_SIZE, BOARD_SIZE))
    if line_score(rect) < 0.08:
        return None
    return quad


def detect_sudoku_quad(frame: np.ndarray, min_area_ratio: float = 0.08) -> np.ndarray | None:
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    th = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 5)
    th = cv.morphologyEx(th, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (5, 5)), iterations=2)

    contours = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]
    frame_area = float(frame.shape[0] * frame.shape[1])
    best_quad = None
    best_score = -1e9

    for cnt in contours:
        area = abs(cv.contourArea(cnt))
        if area < min_area_ratio * frame_area:
            continue
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4 or not cv.isContourConvex(approx):
            continue

        quad = order_quad(approx.reshape(4, 2))
        side_lengths = [
            np.linalg.norm(quad[1] - quad[0]),
            np.linalg.norm(quad[2] - quad[1]),
            np.linalg.norm(quad[3] - quad[2]),
            np.linalg.norm(quad[0] - quad[3]),
        ]
        if min(side_lengths) < 120:
            continue
        ratio = max(side_lengths) / max(1e-6, min(side_lengths))
        if ratio > 1.55:
            continue

        H = cv.getPerspectiveTransform(quad, CANONICAL_QUAD)
        rect = cv.warpPerspective(gray, H, (BOARD_SIZE, BOARD_SIZE))
        grid = line_score(rect)
        rectangularity = area / max(1.0, cv.contourArea(cv.boxPoints(cv.minAreaRect(cnt))))
        score = 5.0 * grid + 2.0 * (area / frame_area) + rectangularity - 0.4 * ratio
        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is not None:
        return best_quad
    return detect_sudoku_quad_hough(gray)


def normalize_digit(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.uint8)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y1:y2, x1:x2]

    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv.resize(crop, (nw, nh), interpolation=cv.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x = (28 - nw) // 2
    y = (28 - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized

    m = cv.moments(canvas)
    if abs(m["m00"]) > 1e-6:
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        M = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        canvas = cv.warpAffine(canvas, M, (28, 28))
    return canvas


def make_hog() -> cv.HOGDescriptor:
    return cv.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)


def hog_feature(hog: cv.HOGDescriptor, img: np.ndarray) -> np.ndarray:
    feat = hog.compute(img).reshape(1, -1).astype(np.float32)
    return feat


def synthetic_digit(label: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((56, 56), dtype=np.uint8)
    fonts = [
        cv.FONT_HERSHEY_SIMPLEX,
        cv.FONT_HERSHEY_DUPLEX,
        cv.FONT_HERSHEY_COMPLEX,
        cv.FONT_HERSHEY_TRIPLEX,
        cv.FONT_HERSHEY_PLAIN,
    ]
    font = int(rng.choice(fonts))
    scale = float(rng.uniform(1.05, 1.85))
    thickness = int(rng.integers(1, 4))
    text = str(label)
    (tw, th), base = cv.getTextSize(text, font, scale, thickness)
    x = int((56 - tw) / 2 + rng.integers(-5, 6))
    y = int((56 + th) / 2 + rng.integers(-5, 6))
    cv.putText(canvas, text, (x, y), font, scale, 255, thickness, cv.LINE_AA)

    angle = float(rng.uniform(-11.0, 11.0))
    tx = float(rng.uniform(-3.0, 3.0))
    ty = float(rng.uniform(-3.0, 3.0))
    M = cv.getRotationMatrix2D((28, 28), angle, float(rng.uniform(0.88, 1.12)))
    M[:, 2] += (tx, ty)
    canvas = cv.warpAffine(canvas, M, (56, 56), flags=cv.INTER_LINEAR, borderValue=0)

    if rng.random() < 0.35:
        canvas = cv.GaussianBlur(canvas, (3, 3), 0)
    if rng.random() < 0.25:
        noise = rng.normal(0, 8, canvas.shape).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    _, canvas = cv.threshold(canvas, 30, 255, cv.THRESH_BINARY)
    return normalize_digit(canvas)


def train_digit_knn(samples_per_digit: int = 650) -> tuple[cv.ml_KNearest, cv.HOGDescriptor]:
    rng = np.random.default_rng(1234)
    hog = make_hog()
    features = []
    labels = []
    for digit in range(1, 10):
        for _ in range(samples_per_digit):
            img = synthetic_digit(digit, rng)
            features.append(hog_feature(hog, img).reshape(-1))
            labels.append(digit)
    train_data = np.asarray(features, dtype=np.float32)
    responses = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    knn = cv.ml.KNearest_create()
    knn.setDefaultK(5)
    knn.train(train_data, cv.ml.ROW_SAMPLE, responses)
    return knn, hog


def remove_grid_lines(rectified_gray: np.ndarray) -> np.ndarray:
    blur = cv.GaussianBlur(rectified_gray, (3, 3), 0)
    bw = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 19, 4)

    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (CELL // 2, 1))
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, CELL // 2))
    horizontal = cv.morphologyEx(bw, cv.MORPH_OPEN, h_kernel, iterations=1)
    vertical = cv.morphologyEx(bw, cv.MORPH_OPEN, v_kernel, iterations=1)
    grid = cv.dilate(cv.bitwise_or(horizontal, vertical), cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)

    clean = cv.bitwise_and(bw, cv.bitwise_not(grid))
    clean = cv.morphologyEx(clean, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)), iterations=1)
    clean = cv.morphologyEx(clean, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)), iterations=1)
    return clean


def extract_cell_digit(clean_mask: np.ndarray, row: int, col: int) -> tuple[np.ndarray | None, float]:
    y0, y1 = row * CELL, (row + 1) * CELL
    x0, x1 = col * CELL, (col + 1) * CELL
    cell = clean_mask[y0:y1, x0:x1]
    margin = 1
    inner = cell[margin : CELL - margin, margin : CELL - margin]

    n, cc, st, _ = cv.connectedComponentsWithStats(inner)
    selected = []
    total_area = 0
    for i in range(1, n):
        area = int(st[i, cv.CC_STAT_AREA])
        w = int(st[i, cv.CC_STAT_WIDTH])
        h = int(st[i, cv.CC_STAT_HEIGHT])
        if area >= 8 and h >= 3 and w >= 2:
            # Rechaza restos muy alargados de la cuadricula.
            if max(w, h) / max(1, min(w, h)) > 9:
                continue
            selected.append(i)
            total_area += area

    ink_ratio = float(total_area / max(1, inner.size))
    if not selected or ink_ratio < 0.012:
        return None, ink_ratio

    mask = np.zeros_like(inner)
    for i in selected:
        mask[cc == i] = 255
    return normalize_digit(mask), ink_ratio


def classify_digit(knn: cv.ml_KNearest, hog: cv.HOGDescriptor, digit_img: np.ndarray) -> tuple[int, float]:
    feat = hog_feature(hog, digit_img)
    _, result, neighbours, dists = knn.findNearest(feat, k=7)
    votes = neighbours.astype(np.int32).reshape(-1)
    values, counts = np.unique(votes, return_counts=True)
    best_idx = int(np.argmax(counts))
    pred = int(values[best_idx])
    vote_conf = float(counts[best_idx] / len(votes))
    dist_conf = float(1.0 / (1.0 + np.mean(dists)))
    conf = 0.75 * vote_conf + 0.25 * min(1.0, dist_conf * 20.0)
    return pred, conf


def rotate_digit(img: np.ndarray, rotation: int) -> np.ndarray:
    rotation %= 4
    if rotation == 0:
        return img
    if rotation == 1:
        return cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    if rotation == 2:
        return cv.rotate(img, cv.ROTATE_180)
    return cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)


def read_grid(
    rectified_gray: np.ndarray,
    knn: cv.ml_KNearest,
    hog: cv.HOGDescriptor,
    min_conf: float,
    digit_rotation: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.zeros((9, 9), dtype=np.int32)
    confs = np.zeros((9, 9), dtype=np.float32)
    digit_debug = np.zeros((9 * 28, 9 * 28), dtype=np.uint8)
    clean_mask = remove_grid_lines(rectified_gray)

    for r in range(9):
        for c in range(9):
            digit_img, ink = extract_cell_digit(clean_mask, r, c)
            if digit_img is None:
                continue
            digit_img = rotate_digit(digit_img, digit_rotation)
            pred, conf = classify_digit(knn, hog, digit_img)
            if conf >= min_conf:
                grid[r, c] = pred
                confs[r, c] = conf
                digit_debug[r * 28 : (r + 1) * 28, c * 28 : (c + 1) * 28] = digit_img
            elif ink > 0.035:
                confs[r, c] = conf

    return grid, confs, digit_debug


def conflict_count(grid: np.ndarray) -> int:
    conflicts = 0
    for i in range(9):
        for vals in (grid[i, :], grid[:, i]):
            nums = [int(x) for x in vals if x != 0]
            conflicts += len(nums) - len(set(nums))
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            nums = [int(x) for x in grid[br : br + 3, bc : bc + 3].reshape(-1) if x != 0]
            conflicts += len(nums) - len(set(nums))
    return conflicts


def conflicting_cells(grid: np.ndarray) -> set[tuple[int, int]]:
    bad: set[tuple[int, int]] = set()

    def mark_duplicates(cells: list[tuple[int, int]]) -> None:
        by_digit: dict[int, list[tuple[int, int]]] = {}
        for r, c in cells:
            d = int(grid[r, c])
            if d != 0:
                by_digit.setdefault(d, []).append((r, c))
        for same in by_digit.values():
            if len(same) > 1:
                bad.update(same)

    for i in range(9):
        mark_duplicates([(i, c) for c in range(9)])
        mark_duplicates([(r, i) for r in range(9)])
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            mark_duplicates([(r, c) for r in range(br, br + 3) for c in range(bc, bc + 3)])
    return bad


def prune_conflicting_digits(grid: np.ndarray, score: np.ndarray) -> np.ndarray:
    pruned = grid.copy()
    safe_score = score.astype(np.float32).copy()
    for _ in range(81):
        bad = conflicting_cells(pruned)
        if not bad:
            break
        victim = min(bad, key=lambda rc: (float(safe_score[rc]), int(pruned[rc])))
        pruned[victim] = 0
        safe_score[victim] = 0.0
    return pruned


def read_grid_auto_orientation(
    rectified_gray: np.ndarray,
    knn: cv.ml_KNearest,
    hog: cv.HOGDescriptor,
    min_conf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    best = None
    best_score = -1e9
    for rotation in range(4):
        grid, confs, digit_debug = read_grid(rectified_gray, knn, hog, min_conf, digit_rotation=rotation)
        filled = int(np.count_nonzero(grid))
        conflicts = conflict_count(grid)
        score = filled - 5 * conflicts
        if valid_grid(grid):
            score += 100
        if score > best_score:
            best_score = score
            best = (grid, confs, digit_debug, rotation)
    assert best is not None
    return best


def stable_grid_from_votes(votes: list[list[deque[int]]], min_votes: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((9, 9), dtype=np.int32)
    support = np.zeros((9, 9), dtype=np.int32)
    for r in range(9):
        for c in range(9):
            vals = [v for v in votes[r][c] if v != 0]
            if not vals:
                continue
            counts = np.bincount(vals, minlength=10)
            digit = int(np.argmax(counts))
            if counts[digit] >= min_votes:
                grid[r, c] = digit
                support[r, c] = int(counts[digit])
    return grid, support


def valid_grid(grid: np.ndarray) -> bool:
    for i in range(9):
        row = [int(x) for x in grid[i, :] if x != 0]
        col = [int(x) for x in grid[:, i] if x != 0]
        if len(row) != len(set(row)) or len(col) != len(set(col)):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = [int(x) for x in grid[br : br + 3, bc : bc + 3].reshape(-1) if x != 0]
            if len(block) != len(set(block)):
                return False
    return True


def solve_sudoku(grid: np.ndarray) -> np.ndarray | None:
    board = grid.astype(np.int32).copy()

    def candidates(r: int, c: int) -> list[int]:
        used = set(int(x) for x in board[r, :] if x != 0)
        used.update(int(x) for x in board[:, c] if x != 0)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        used.update(int(x) for x in board[br : br + 3, bc : bc + 3].reshape(-1) if x != 0)
        return [d for d in range(1, 10) if d not in used]

    def backtrack() -> bool:
        best = None
        best_cands = None
        for r in range(9):
            for c in range(9):
                if board[r, c] == 0:
                    cand = candidates(r, c)
                    if not cand:
                        return False
                    if best_cands is None or len(cand) < len(best_cands):
                        best = (r, c)
                        best_cands = cand
        if best is None:
            return True
        r, c = best
        for d in best_cands:
            board[r, c] = d
            if backtrack():
                return True
            board[r, c] = 0
        return False

    if not valid_grid(board):
        return None
    if backtrack():
        return board
    return None


def draw_board_debug(rectified: np.ndarray, grid: np.ndarray, confs: np.ndarray, support: np.ndarray) -> np.ndarray:
    vis = cv.cvtColor(rectified, cv.COLOR_GRAY2BGR)
    for i in range(10):
        p = i * CELL
        thick = 2 if i % 3 == 0 else 1
        cv.line(vis, (p, 0), (p, BOARD_SIZE), (0, 180, 255), thick, cv.LINE_AA)
        cv.line(vis, (0, p), (BOARD_SIZE, p), (0, 180, 255), thick, cv.LINE_AA)
    for r in range(9):
        for c in range(9):
            d = int(grid[r, c])
            if d == 0:
                continue
            x = c * CELL + 16
            y = r * CELL + 34
            cv.putText(vis, str(d), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(vis, f"{confs[r,c]:.2f}/{support[r,c]}", (c * CELL + 3, r * CELL + 48), cv.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv.LINE_AA)
    return vis


def draw_solution_overlay(solution: np.ndarray, givens: np.ndarray) -> np.ndarray:
    overlay = np.zeros((BOARD_SIZE, BOARD_SIZE, 4), dtype=np.uint8)
    for r in range(9):
        for c in range(9):
            if givens[r, c] != 0:
                continue
            value = int(solution[r, c])
            if value == 0:
                continue
            text = str(value)
            font = cv.FONT_HERSHEY_SIMPLEX
            scale = 1.15
            thick = 2
            (tw, th), _ = cv.getTextSize(text, font, scale, thick)
            x = c * CELL + (CELL - tw) // 2
            y = r * CELL + (CELL + th) // 2
            cv.putText(overlay, text, (x, y), font, scale, (255, 80, 20, 230), thick, cv.LINE_AA)
    return overlay


def warp_alpha_overlay(frame: np.ndarray, overlay: np.ndarray, H_inv: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    warped = cv.warpPerspective(overlay, H_inv, (w, h))
    alpha = warped[:, :, 3].astype(np.float32) / 255.0
    out = frame.astype(np.float32)
    for ch in range(3):
        out[:, :, ch] = out[:, :, ch] * (1.0 - alpha) + warped[:, :, ch].astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def grid_to_text(grid: np.ndarray) -> str:
    return "\n".join("".join(str(int(x)) if x else "." for x in row) for row in grid)


def grid_to_pretty_text(grid: np.ndarray) -> str:
    rows = []
    for r in range(9):
        vals = []
        for c in range(9):
            vals.append(str(int(grid[r, c])) if grid[r, c] else ".")
            if c in (2, 5):
                vals.append("|")
        rows.append(" ".join(vals))
        if r in (2, 5):
            rows.append("-" * 21)
    return "\n".join(rows)


def save_diagnostics(
    save_dir: Path,
    stamp: str,
    vis: np.ndarray,
    rect_gray: np.ndarray | None,
    current_grid: np.ndarray,
    stable_grid: np.ndarray,
    cleaned_grid: np.ndarray,
    support: np.ndarray,
    solution: np.ndarray | None,
    confs: np.ndarray,
) -> None:
    cv.imwrite(str(save_dir / f"sudoku_ra_{stamp}.png"), vis)
    if rect_gray is not None:
        cv.imwrite(str(save_dir / f"sudoku_rectificado_{stamp}.png"), rect_gray)
        debug = draw_board_debug(rect_gray, cleaned_grid, confs, support)
        cv.imwrite(str(save_dir / f"sudoku_ocr_{stamp}.png"), debug)

    lines = [
        "LECTURA OCR DEL FRAME",
        grid_to_pretty_text(current_grid),
        "",
        "LECTURA ESTABILIZADA",
        grid_to_pretty_text(stable_grid),
        "",
        "TABLERO USADO PARA RESOLVER",
        grid_to_pretty_text(cleaned_grid),
        "",
        f"valido={valid_grid(cleaned_grid)}",
        f"pistas={int(np.count_nonzero(cleaned_grid))}",
        "",
    ]
    if solution is None:
        lines.extend(["SOLUCION", "No resuelto."])
    else:
        lines.extend(["SOLUCION", grid_to_pretty_text(solution), "", f"solucion_valida={valid_grid(solution)}"])
    (save_dir / f"sudoku_diagnostico_{stamp}.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sudoku en realidad aumentada con OCR en vivo")
    add_stream_args(parser)
    parser.add_argument("--min-area-ratio", type=float, default=0.08, help="area minima del tablero respecto al frame")
    parser.add_argument("--ocr-conf", type=float, default=0.80, help="confianza minima para aceptar un digito OCR")
    parser.add_argument("--vote-window", type=int, default=10, help="numero de frames usados para voto temporal")
    parser.add_argument("--min-votes", type=int, default=4, help="votos necesarios para fijar un digito")
    parser.add_argument("--min-givens", type=int, default=17, help="minimo de pistas leidas antes de resolver")
    parser.add_argument("--smooth", type=float, default=0.35, help="suavizado temporal del cuadrilatero")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True, help="muestra ventanas de depuracion OCR")
    parser.add_argument("--save-dir", default=str(Path(__file__).with_name("captures")), help="carpeta para capturas con tecla S")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Entrenando clasificador sintetico de digitos...")
    knn, hog = train_digit_knn()
    print("Listo. Controles: R reinicia lectura | S guarda captura | D debug on/off | Q o ESC sale")

    votes = [[deque(maxlen=int(args.vote_window)) for _ in range(9)] for _ in range(9)]
    stable_quad: np.ndarray | None = None
    solution: np.ndarray | None = None
    solution_givens: np.ndarray | None = None
    fps_est = 25.0
    prev_t = None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for key, frame in get_stream(args):
        now = time.time()
        if prev_t is None:
            prev_t = now
        dt = max(1e-3, now - prev_t)
        prev_t = now
        fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

        if key in (27, ord("q")):
            break
        if key == ord("d"):
            args.debug = not bool(args.debug)
        if key == ord("r"):
            votes = [[deque(maxlen=int(args.vote_window)) for _ in range(9)] for _ in range(9)]
            stable_quad = None
            solution = None
            solution_givens = None

        vis = frame.copy()
        status = "SEARCH"
        detected = detect_sudoku_quad(frame, float(args.min_area_ratio))
        rect_gray = None
        current_grid = np.zeros((9, 9), dtype=np.int32)
        stable_grid = np.zeros((9, 9), dtype=np.int32)
        cleaned_grid = np.zeros((9, 9), dtype=np.int32)
        confs = np.zeros((9, 9), dtype=np.float32)
        support = np.zeros((9, 9), dtype=np.int32)

        if detected is not None:
            stable_quad = smooth_quad(stable_quad, detected, float(args.smooth))
            cv.polylines(vis, [np.round(stable_quad).astype(np.int32)], True, (0, 220, 255), 2, cv.LINE_AA)

            H = cv.getPerspectiveTransform(stable_quad, CANONICAL_QUAD)
            H_inv = cv.getPerspectiveTransform(CANONICAL_QUAD, stable_quad)
            rect = cv.warpPerspective(frame, H, (BOARD_SIZE, BOARD_SIZE))
            rect_gray = cv.cvtColor(rect, cv.COLOR_BGR2GRAY)
            current_grid, confs, _, rotation = read_grid_auto_orientation(rect_gray, knn, hog, float(args.ocr_conf))

            for r in range(9):
                for c in range(9):
                    votes[r][c].append(int(current_grid[r, c]))

            stable_grid, support = stable_grid_from_votes(votes, int(args.min_votes))
            cleaned_grid = prune_conflicting_digits(stable_grid, support)
            if valid_grid(cleaned_grid):
                filled = int(np.count_nonzero(cleaned_grid))
                status = f"READING {filled}/81"
                if filled >= int(args.min_givens):
                    solved = solve_sudoku(cleaned_grid)
                    if solved is not None:
                        solution = solved
                        solution_givens = cleaned_grid.copy()
                        status = "SOLVED"
            else:
                status = "INVALID OCR"
                solution = None
                solution_givens = None

            if solution is not None and solution_givens is not None:
                overlay = draw_solution_overlay(solution, solution_givens)
                vis = warp_alpha_overlay(vis, overlay, H_inv)
                status = "SOLVED"

            if args.debug:
                dbg = draw_board_debug(rect_gray, cleaned_grid, confs, support)
                cv.putText(dbg, f"rot={rotation * 90} deg", (10, BOARD_SIZE - 12), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv.LINE_AA)
                cv.imshow("Sudoku OCR rectificado", dbg)
        cv.putText(vis, f"{status}  fps={fps_est:.1f}", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(vis, "R reinicia | S captura | D debug | Q/ESC sale", (10, vis.shape[0] - 14), cv.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv.LINE_AA)

        if key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            save_diagnostics(save_dir, stamp, vis, rect_gray, current_grid, stable_grid, cleaned_grid, support, solution, confs)
            print(f"Diagnostico guardado en: {save_dir}")

        cv.imshow("RA Sudoku", vis)

    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
