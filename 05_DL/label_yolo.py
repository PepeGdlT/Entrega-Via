#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class BoxEditor:
    def __init__(self, window_name: str, image_shape, scale: float):
        self.window_name = window_name
        self.img_h, self.img_w = image_shape[:2]
        self.scale = scale
        self.boxes: list[tuple[int, int, int, int]] = []
        self.dragging = False
        self.start: tuple[int, int] | None = None
        self.current: tuple[int, int] | None = None
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _clip_orig(self, x: int, y: int) -> tuple[int, int]:
        x = max(0, min(x, self.img_w - 1))
        y = max(0, min(y, self.img_h - 1))
        return x, y

    def _display_to_orig(self, x: int, y: int) -> tuple[int, int]:
        ox = int(round(x / self.scale))
        oy = int(round(y / self.scale))
        return self._clip_orig(ox, oy)

    def _on_mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = self._display_to_orig(x, y)
            self.current = self.start
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.current = self._display_to_orig(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.current = self._display_to_orig(x, y)
            if self.start is None or self.current is None:
                return
            x1, y1 = self.start
            x2, y2 = self.current
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if (x2 - x1) >= 6 and (y2 - y1) >= 6:
                self.boxes.append((x1, y1, x2, y2))
            self.start = None
            self.current = None

    def undo(self) -> None:
        if self.boxes:
            self.boxes.pop()

    def clear(self) -> None:
        self.boxes.clear()
        self.dragging = False
        self.start = None
        self.current = None

    def draw(self, image):
        vis = image.copy()
        for x1, y1, x2, y2 in self.boxes:
            p1 = (int(round(x1 * self.scale)), int(round(y1 * self.scale)))
            p2 = (int(round(x2 * self.scale)), int(round(y2 * self.scale)))
            cv2.rectangle(vis, p1, p2, (0, 255, 0), 2)
        if self.dragging and self.start is not None and self.current is not None:
            x1, y1 = self.start
            x2, y2 = self.current
            p1 = (int(round(x1 * self.scale)), int(round(y1 * self.scale)))
            p2 = (int(round(x2 * self.scale)), int(round(y2 * self.scale)))
            cv2.rectangle(vis, p1, p2, (0, 200, 255), 2)
        return vis


def list_pending_images(source_dir: Path, dataset_dir: Path) -> list[Path]:
    labeled_names = set()
    for split in ("train", "val"):
        images_dir = dataset_dir / split / "images"
        if images_dir.exists():
            for path in images_dir.iterdir():
                if path.is_file():
                    labeled_names.add(path.name)

    pending = []
    for path in sorted(source_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS and path.name not in labeled_names:
            pending.append(path)
    return pending


def suggested_split(image_path: Path, val_ratio: float) -> str:
    digest = hashlib.sha1(image_path.name.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return "val" if value < val_ratio else "train"


def ensure_dataset_dirs(dataset_dir: Path) -> None:
    for split in ("train", "val"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def write_yolo_label(label_path: Path, image_shape, boxes, class_id: int) -> None:
    h, w = image_shape[:2]
    with label_path.open("w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            xc = ((x1 + x2) * 0.5) / w
            yc = ((y1 + y2) * 0.5) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def scale_for_display(image_shape, max_width: int, max_height: int) -> float:
    h, w = image_shape[:2]
    return min(max_width / float(w), max_height / float(h), 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Etiquetado YOLO simple para la clase taza")
    parser.add_argument("--source", default=str(Path(__file__).with_name("captures")), help="imagenes sin etiquetar")
    parser.add_argument("--dataset", default=str(Path(__file__).with_name("dataset")), help="dataset YOLO destino")
    parser.add_argument("--class-id", type=int, default=0, help="id numerico de la clase")
    parser.add_argument("--label", default="taza", help="nombre de clase mostrado en pantalla")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="porcentaje aproximado reservado a validacion")
    parser.add_argument("--max-width", type=int, default=1280, help="ancho maximo de vista")
    parser.add_argument("--max-height", type=int, default=900, help="alto maximo de vista")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source)
    dataset_dir = Path(args.dataset)
    ensure_dataset_dirs(dataset_dir)

    if not source_dir.exists():
        raise SystemExit(f"No existe el directorio fuente: {source_dir}")

    pending = list_pending_images(source_dir, dataset_dir)
    if not pending:
        print("No hay imagenes pendientes de etiquetar.")
        return 0

    window_name = "Etiquetado YOLO"
    cv2.namedWindow(window_name)

    print("Controles: arrastra para crear caja, U deshace, C limpia, T/V elige split, ENTER guarda, X salta, Q sale.")

    idx = 0
    while idx < len(pending):
        image_path = pending[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"No se pudo abrir: {image_path}")
            idx += 1
            continue

        split = suggested_split(image_path, args.val_ratio)
        scale = scale_for_display(image.shape, args.max_width, args.max_height)
        disp = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image
        editor = BoxEditor(window_name, image.shape, scale)

        while True:
            vis = editor.draw(disp)
            cv2.putText(
                vis,
                f"{idx + 1}/{len(pending)} {image_path.name} | class={args.label} | split={split} | boxes={len(editor.boxes)}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                vis,
                "ENTER=save  T=train  V=val  U=undo  C=clear  X=skip  Q=quit",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (230, 230, 230),
                1,
            )
            cv2.imshow(window_name, vis)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("u"):
                editor.undo()
            elif key == ord("c"):
                editor.clear()
            elif key == ord("t"):
                split = "train"
            elif key == ord("v"):
                split = "val"
            elif key in (13, ord("s")):
                img_out = dataset_dir / split / "images" / image_path.name
                lbl_out = dataset_dir / split / "labels" / f"{image_path.stem}.txt"
                shutil.copy2(image_path, img_out)
                write_yolo_label(lbl_out, image.shape, editor.boxes, args.class_id)
                print(f"Guardado {split}: {img_out.name} con {len(editor.boxes)} cajas")
                idx += 1
                break
            elif key == ord("x"):
                print(f"Saltada: {image_path.name}")
                idx += 1
                break
            elif key in (27, ord("q")):
                cv2.destroyAllWindows()
                print("Etiquetado interrumpido.")
                return 0

    cv2.destroyAllWindows()
    print("Etiquetado completado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
