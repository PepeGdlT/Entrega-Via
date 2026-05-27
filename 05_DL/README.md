# 05 - DL con YOLO (`taza`)

Ejercicio de deep learning basado en el flujo de `umucv/code/DL/yolotrain`, adaptado a una clase propia: `taza`.

## Estructura

```text
05_DL/
  captures/              # imagenes crudas capturadas con webcam
  dataset/
    train/
      images/
      labels/
    val/
      images/
      labels/
  models/
    taza.pt              # modelo final copiado aqui tras entrenar
  capture_taza.py        # captura de imagenes
  label_yolo.py          # etiquetado manual en formato YOLO
  train_taza.py          # entrenamiento con Ultralytics
  run_model.py           # prueba en vivo
  taza.yaml              # configuracion del dataset
```

## 1) Instalar dependencias

Desde la raiz del proyecto:

```powershell
pip install -r .\05_DL\requirements.txt
```

Si ya tienes `ultralytics` y OpenCV por otras practicas, no hace falta repetirlo.

## 2) Capturar imagenes de tu taza

```powershell
python .\05_DL\capture_taza.py --dev 0
```

Controles:

- `S`: guarda el frame actual
- `A`: activa/desactiva autocaptura
- `Q` o `ESC`: salir

Recomendacion:

- Captura entre `60` y `120` imagenes.
- Cambia distancia, orientacion, iluminacion y fondo.
- Incluye algunos casos algo mas dificiles: taza parcialmente tapada, inclinada o cerca del borde.

## 3) Etiquetar las cajas

```powershell
python .\05_DL\label_yolo.py
```

Controles:

- Arrastra con el raton para dibujar una caja.
- `U`: deshacer ultima caja.
- `C`: limpiar cajas.
- `T`: guardar la imagen en `train`.
- `V`: guardar la imagen en `val`.
- `ENTER`: guardar la imagen y su `.txt`.
- `X`: saltar esa imagen.
- `Q` o `ESC`: salir.

Notas:

- Si guardas una imagen sin cajas, se crea una etiqueta vacia. Puede servir como ejemplo negativo.
- El script propone `train/val` automaticamente con `80/20`, pero puedes cambiarlo manualmente con `T` o `V`.

## 4) Entrenar el detector

```powershell
python .\05_DL\train_taza.py --epochs 100 --imgsz 640
```

El script:

- usa `taza.yaml`
- entrena con `yolo11n.pt`
- guarda resultados en `05_DL/runs/`
- copia el mejor modelo a `05_DL/models/taza.pt`

Si quieres usar exactamente el estilo del ejemplo del profesor, el comando equivalente seria:

```powershell
yolo detect train data=.\05_DL\taza.yaml model=yolo11n.pt epochs=100 imgsz=640 augment=True
```

## 5) Probar el modelo entrenado

```powershell
python .\05_DL\run_model.py --dev 0 --conf 0.35
```

Tambien puedes probar con un video:

```powershell
python .\05_DL\run_model.py --dev .\ruta\video.mp4
```

## Que conviene entregar

- Capturas de tu dataset.
- Un ejemplo de etiqueta YOLO (`.txt`).
- `taza.yaml`.
- El comando usado para entrenar.
- Graficas o capturas de `runs/`.
- Ejemplos del detector funcionando.
- El modelo final `models/taza.pt`.

## Relacion con `yolotrain` del profesor

Se reutiliza la misma idea base:

- estructura `train/images`, `train/labels`, `val/images`, `val/labels`
- archivo `.yaml` con clases
- entrenamiento con Ultralytics
- prueba final con un script de inferencia

La diferencia es que aqui no usamos `facelabel.py`, porque ese ejemplo automatiza cajas para boca. Para `taza`, el etiquetado se hace manualmente con `label_yolo.py`.
