# Actividad 03 - TownCentre (ROI + movimiento + YOLOv8 + Telegram)

Pipeline implementado en `actividad.py`:

1. Carga `TownCentre.mp4`.
2. Te deja seleccionar ROI manual en el primer frame y no empieza hasta pulsar `ENTER`/`SPACE`.
3. Durante ejecucion puedes activar modo edicion ROI con tecla `r`.
4. Detecta movimiento **solo** dentro de la ROI (MOG2).
5. Si hay movimiento, ejecuta YOLOv8 para clase `person`.
6. Filtra personas con color rojo en HSV (global + zona central + rojo en pixeles en movimiento).
7. Si hay evento, anonimiza (blur) con persistencia temporal para evitar parpadeos.
8. Guarda imagen + clip de 2-3 s y opcionalmente envia Telegram.

## Requisitos

- Python 3.10+ (recomendado)
- Dependencias de `requirements.txt`

Instalacion (desde la raiz del workspace):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r .\03_Actividad\requirements.txt
```

## Descargar TownCentre.mp4

Opciones practicas (el dataset tiene varios mirrors y alguno puede caer temporalmente):

1. **Fuente oficial Oxford Town Centre** (busca `TownCentreXVID.avi` en la pagina del dataset):
   - https://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html
2. **Mirror en GitHub/Kaggle**: busca por `TownCentre.mp4` o `TownCentreXVID.avi`.

Cuando lo tengas, dejalo en:

- `03_Actividad/videos/TownCentre.mp4`

Si el archivo se llama distinto (por ejemplo `TownCentreXVID.avi`), puedes:

- renombrarlo a `TownCentre.mp4`, o
- pasarlo por parametro `--video`.

## Telegram: token y chat id

### 1) Crear bot y sacar token

1. Abre Telegram y busca `@BotFather`.
2. Ejecuta `/newbot`.
3. Pon nombre y username.
4. BotFather te devolvera un token tipo `123456:ABC-DEF...`.

### 2) Obtener tu chat id

1. Envia cualquier mensaje a tu bot (por ejemplo `hola`).
2. Abre en navegador (cambia `TOKEN`):
   - `https://api.telegram.org/botTOKEN/getUpdates`
3. En la respuesta JSON busca `"chat":{"id":...}`.

## Ejecucion

Comando recomendado (desde raiz):

```powershell
python -u .\03_Actividad\actividad.py --video .\03_Actividad\videos\TownCentre.mp4 --event-seconds 3 --prebuffer-seconds 1 --cooldown-seconds 5 --conf 0.35 --dnn-every 2
```

Con Telegram:

```powershell
$env:TELEGRAM_BOT_TOKEN="TU_TOKEN"
$env:TELEGRAM_CHAT_ID="TU_CHAT_ID"
python -u .\03_Actividad\actividad.py --video .\03_Actividad\videos\TownCentre.mp4
```

Sin Telegram:

```powershell
python -u .\03_Actividad\actividad.py --video .\03_Actividad\videos\TownCentre.mp4 --no-telegram
```

## Parametros importantes

- `--motion-area-min`: area minima de movimiento en ROI (si no detecta nada, bajalo)
- `--motion-thr`: umbral binario de mascara de movimiento
- `--red-ratio-thr`: porcentaje de rojo minimo para disparar evento
- `--red-center-ratio-thr`: minimo de rojo en zona central del cuerpo
- `--red-motion-ratio-thr`: rojo minimo sobre partes en movimiento (reduce escaparates)
- `--red-center-scale`: tamano del recorte central para validar rojo
- `--anon-hold-frames`: mantiene blur unos frames si YOLO pierde deteccion
- `--track-iou-thr`: sensibilidad para mantener la misma persona entre frames
- `--dnn-every`: YOLO cada N frames (subirlo mejora rendimiento)
- `--input-size`: resolucion de entrada YOLO (`640` por calidad, `416/320` por velocidad)

## Ajuste en vivo con barras

Ahora los parametros de deteccion se pueden cambiar durante la ejecucion con barras en la ventana principal.
Los argumentos CLI siguen existiendo, pero actuan como valor inicial de cada barra.

Barras recomendadas:

- `conf x100`: confianza minima de YOLO. Mas bajo detecta mas (y mete mas ruido).
- `dnn every`: ejecuta YOLO cada N frames. `1` es mas preciso; `2-3` mas rapido.
- `motion thr`: umbral de mascara de movimiento. Mas bajo = mas sensible.
- `motion area min`: area minima de movimiento. Si no detecta casi nada, bajala.
- `red ratio x100`: porcentaje rojo minimo en bbox completa.
- `red center x100`: rojo minimo en zona central de persona (filtra fondo rojo lateral).
- `red motion x1000`: rojo minimo que ademas coincide con pixeles en movimiento.
- `red s min` y `red v min`: saturacion y brillo minimos para considerar un pixel rojo.
- `red center scale x100`: tamano del recorte central (65 suele ir bien).
- `anon hold frames`: mantiene blur unos frames cuando YOLO falla momentaneamente.
- `track iou x100`: exigencia para enlazar la misma persona entre frames.

Flujo de ajuste rapido:

1. Primero ajusta movimiento: `motion thr` y `motion area min`.
2. Luego ajusta YOLO: `conf` y `dnn every`.
3. Finalmente afina el rojo: `red ratio`, `red center`, `red motion`.

## Controles de teclado

- `ENTER` / `SPACE`: iniciar tras dibujar ROI inicial
- `r`: activar/desactivar modo edicion ROI durante ejecucion
- `x`: limpiar ROI actual
- `q` o `ESC`: salir

## Salidas

Todo se guarda en:

- `03_Actividad/eventos/`

Por cada evento:

- `evento_YYYYMMDD_HHMMSS.jpg` (frame anonimizado para aviso)
- `evento_YYYYMMDD_HHMMSS.mp4` (clip con blur)

## Pruebas rapidas

```powershell
python -u .\03_Actividad\actividad.py --self-test
python -u .\03_Actividad\test_actividad.py
```
