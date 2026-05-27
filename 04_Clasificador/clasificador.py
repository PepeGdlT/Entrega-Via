from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from core.model_store import ModelStore
from methods.registry import available_methods, create_method


WINDOW_NAME = "Clasificador"


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
		parser.add_argument("--dev", type=str, default="0", help="fuente de video")
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


def parse_args():
	parser = argparse.ArgumentParser(description="Clasificador modular de imagenes en tiempo real")
	_add_stream_args(parser)
	parser.add_argument("--models", required=True, help="directorio con modelos por subcarpeta")
	parser.add_argument("--method", required=True, choices=available_methods(), help="metodo de comparacion")
	parser.add_argument("--topk", type=int, default=5, help="numero de resultados a mostrar")
	parser.add_argument("--every", type=int, default=2, help="procesar 1 de cada N frames")
	parser.add_argument("--min-score", type=float, default=0.0, help="umbral minimo para aceptar resultado")
	parser.add_argument("--save-added", action="store_true", help="guardar en disco modelos anadidos en caliente")
	parser.add_argument("--embedder-model", default=None, help="ruta .tflite para mediapipe_embedding")
	parser.add_argument("--sift-ratio", type=float, default=0.75, help="ratio test de Lowe para SIFT")
	args, rest = parser.parse_known_args(sys.argv)
	unknown = [p for p in rest[1:] if p and p.strip() and p != "`"]
	if unknown:
		parser.error("unknown parameters: " + str(unknown))
	return args


def _crop_with_roi(frame, roi_box):
	if roi_box is None:
		return frame
	x1, y1, x2, y2 = roi_box
	h, w = frame.shape[:2]
	x1 = max(0, min(x1, w - 1))
	y1 = max(0, min(y1, h - 1))
	x2 = max(0, min(x2, w - 1))
	y2 = max(0, min(y2, h - 1))
	if x2 <= x1 or y2 <= y1:
		return frame
	return frame[y1 : y2 + 1, x1 : x2 + 1]


def _save_runtime_model(image_bgr, models_dir: str, label: str) -> str:
	safe_label = label.strip().replace(" ", "_")
	label_dir = Path(models_dir) / safe_label
	label_dir.mkdir(parents=True, exist_ok=True)
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	out_path = label_dir / f"runtime_{stamp}.jpg"
	cv2.imwrite(str(out_path), image_bgr)
	return str(out_path)


def _draw_hud(frame, roi_box, method_name, loaded_models, fps_est, result, min_score, format_score):
	y = 22
	cv2.putText(frame, f"method={method_name} models={loaded_models} fps={fps_est:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
	y += 24
	cv2.putText(frame, "R:ROI  X:limpiar ROI  N:nuevo modelo  Q:salir", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
	y += 24

	if roi_box is not None:
		x1, y1, x2, y2 = roi_box
		cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

	if result is None or result.best_hit is None:
		cv2.putText(frame, "Sin clasificacion", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 255), 2)
		return

	best = result.best_hit
	ok = best.score >= min_score
	color = (0, 220, 0) if ok else (0, 160, 255)
	txt = f"BEST: {best.label}  {format_score(best.score)}  conf={result.confidence:.3f}"
	cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
	y += 24

	for hit in result.top_hits[:4]:
		line = f"- {hit.label:<16} {format_score(hit.score)}"
		cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
		y += 20


def main():
	args = parse_args()

	method_impl = create_method(args.method, args)
	store = ModelStore(method_impl)
	loaded = store.load_from_directory(args.models)
	print(f"Metodo: {args.method}")
	print(f"Modelos precalculados: {loaded}")
	if loaded == 0:
		print("Aviso: no se cargaron modelos validos. Revisa --models y el metodo elegido.")

	cv2.namedWindow(WINDOW_NAME)
	_ensure_umucv_package()
	try:
		from umucv.util import ROI

		region = ROI(WINDOW_NAME)
		roi_enabled = True
	except Exception:
		region = None
		roi_enabled = False

	stream = iter(_get_stream(args))

	fps_est = 25.0
	t_prev = time.time()
	frame_idx = 0
	last_result = None
	roi_edit_mode = False

	for key, frame in stream:
		frame_idx += 1
		t_now = time.time()
		dt = max(1e-3, t_now - t_prev)
		t_prev = t_now
		fps_est = 0.92 * fps_est + 0.08 * (1.0 / dt)

		if key == ord("r"):
			roi_edit_mode = not roi_edit_mode
		if key in (27, ord("q")):
			break

		roi_box = None
		if roi_enabled and getattr(region, "roi", None):
			if len(region.roi) == 4:
				x1, y1, x2, y2 = region.roi
				x1, x2 = sorted((int(x1), int(x2)))
				y1, y2 = sorted((int(y1), int(y2)))
				if (x2 - x1) > 5 and (y2 - y1) > 5:
					roi_box = (x1, y1, x2, y2)

		if key == ord("x") and roi_enabled:
			region.roi = []
			roi_box = None

		query = _crop_with_roi(frame, roi_box)
		if query is not None and query.size > 0 and (frame_idx % max(1, args.every) == 0):
			last_result = store.classify(query, topk=args.topk)

		if key == ord("n"):
			print("Etiqueta nueva (vacio para cancelar): ", end="", flush=True)
			label = input().strip()
			if label:
				save_path = None
				if args.save_added:
					save_path = _save_runtime_model(query, args.models, label)
				ok = store.add_model_image(query, label=label, save_path=save_path)
				print(f"Modelo anadido: {ok} label={label} path={save_path if save_path else '<memoria>'}")

		vis = frame.copy()
		_draw_hud(
			vis,
			roi_box=roi_box,
			method_name=args.method,
			loaded_models=len(store.entries),
			fps_est=fps_est,
			result=last_result,
			min_score=float(args.min_score),
			format_score=method_impl.format_score,
		)
		if roi_edit_mode:
			cv2.putText(vis, "ROI edit ON (arrastra raton)", (10, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

		cv2.imshow(WINDOW_NAME, vis)

	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()

