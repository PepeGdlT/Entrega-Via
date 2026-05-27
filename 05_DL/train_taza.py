#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def resolve_yaml_path(data_yaml: Path) -> Path:
    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", "."))
    if not root.is_absolute():
        data["path"] = str((data_yaml.parent / root).resolve())

    resolved_yaml = data_yaml.with_name("_dataset_resolved.yaml")
    with resolved_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)
    return resolved_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el detector YOLO de taza")
    parser.add_argument("--data", default=str(BASE_DIR / "taza.yaml"), help="fichero dataset yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="modelo base de Ultralytics")
    parser.add_argument("--epochs", type=int, default=100, help="numero de epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="tamano de entrada")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--patience", type=int, default=30, help="early stopping")
    parser.add_argument("--workers", type=int, default=0, help="workers del dataloader")
    parser.add_argument("--device", default="", help="cpu, 0, 0,1... vacio = auto")
    parser.add_argument("--project", default=str(BASE_DIR / "runs"), help="directorio de salidas")
    parser.add_argument("--name", default="taza", help="nombre de la corrida")
    parser.add_argument("--out-model", default=str(BASE_DIR / "models" / "taza.pt"), help="modelo final copiado")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="data augmentation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise SystemExit(f"No existe el dataset yaml: {data_yaml}")

    resolved_yaml = resolve_yaml_path(data_yaml)
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=str(resolved_yaml),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        patience=int(args.patience),
        workers=int(args.workers),
        augment=bool(args.augment),
        project=str(Path(args.project).resolve()),
        name=args.name,
        exist_ok=True,
        device=(args.device if args.device else None),
    )

    save_dir = None
    if hasattr(results, "save_dir"):
        save_dir = Path(results.save_dir)
    elif getattr(model, "trainer", None) is not None and hasattr(model.trainer, "save_dir"):
        save_dir = Path(model.trainer.save_dir)

    if save_dir is None:
        raise SystemExit("No se pudo localizar la carpeta de entrenamiento de Ultralytics.")

    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise SystemExit(f"No se encontro el modelo entrenado: {best_pt}")

    shutil.copy2(best_pt, out_model)
    print(f"Entrenamiento finalizado.")
    print(f"Resultados: {save_dir}")
    print(f"Modelo final copiado a: {out_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
