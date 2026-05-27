# 04_Clasificador

Aplicacion de clasificacion de imagenes en tiempo real con arquitectura modular de metodos.

## Requisitos

- Python 3.10+
- Entorno virtual activo
- Dependencias en `requirements.txt`

## Estructura esperada de modelos

El directorio pasado en `--models` debe tener subcarpetas por clase:

```text
models/
  embed/
    mobilenet_v3_small.tflite
  libro/
    img1.jpg
    img2.jpg
  cuadro/
    m1.png
  mano_ok/
    hand1.jpg
```

La etiqueta se toma del nombre de subcarpeta.

## Instalacion

```powershell
pip install -r .\04_Clasificador\requirements.txt
```

## Ejecucion rapida

### 1) SIFT (objetos con textura)

```powershell
python .\04_Clasificador\clasificador.py --dev .\03_Actividad\videos\TownCentre.mp4 --models .\04_Clasificador\models --method sift_matching --every 2 --topk 5 --save-added
```

### 2) Gestos de mano (Procrustes)

```powershell
python .\04_Clasificador\clasificador.py --dev 0 --models .\04_Clasificador\models --method hand_procrustes --every 1 --topk 3 --save-added
```

### 3) Embedding MediaPipe

```powershell
python .\04_Clasificador\clasificador.py --dev 0 --models .\04_Clasificador\models --method mediapipe_embedding --every 2 --topk 5 --embedder-model .\04_Clasificador\models\embed\mobilenet_v3_small.tflite --save-added
```

Si no indicas `--embedder-model`, el metodo intenta descargar el modelo automaticamente.

## Anadir varias clases (raton, taza, movil, ...)

- No tienes que entrenar un modelo distinto por clase en `mediapipe_embedding`.
- El `.tflite` es el extractor comun; cada clase se representa con imagenes en su subcarpeta.
- Ejemplo: `models/raton/`, `models/taza/`, `models/movil/` con varias fotos por objeto.
- En ejecucion, pulsa `N` para anadir una muestra nueva y escribe la etiqueta de clase.
- Recomendacion: 5-15 imagenes por clase, variando distancia, angulo y luz.

## Controles en ejecucion

- `R`: activar/desactivar modo edicion ROI (arrastra raton en la ventana).
- `X`: limpiar ROI.
- `N`: anadir modelo en caliente (pregunta etiqueta por consola).
- `Q` o `ESC`: salir.

## Salida mostrada

- Mejor etiqueta (`BEST`), score del metodo y confianza (margen entre top-1 y top-2).
- Top resultados y sus scores.
- FPS aproximado y numero de modelos cargados.

## Test rapido

```powershell
python -m unittest .\04_Clasificador\test_clasificador.py -v
```

## Como anadir un metodo nuevo

1. Crear un archivo en `methods/` con una clase que implemente `build_descriptor` y `score`.
2. Registrar el metodo en `methods/registry.py`.
3. (Opcional) anadir argumentos en `clasificador.py`.

