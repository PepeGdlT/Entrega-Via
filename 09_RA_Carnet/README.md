# 09 - RA Carnet

Sustitucion automatica de la foto del carnet mediante realidad aumentada plana, con plantilla capturada desde la propia app.

## Idea

El flujo intenta aproximarse al estilo del demo del profesor, pero resolviendo la posicion exacta de la foto a partir de una plantilla real del carnet:

1. la app detecta automaticamente el carnet como cuadrilatero;
2. el usuario pulsa `T` cuando el carnet esta frontal y bien visible;
3. la app rectifica esa vista y muestra el carnet en frontal;
4. el usuario selecciona con el raton la zona exacta de la foto en esa plantilla rectificada;
5. a partir de ahi, en tiempo real, la app vuelve a detectar el carnet en cada frame;
6. proyecta la nueva imagen justo sobre la region de la foto, sin clicks ni tracks visibles durante el uso normal.

## Ejecucion

Con webcam:

```powershell
python .\09_RA_Carnet\ra_carnet.py --dev 0 --replace .\ruta\mi_foto.png
```

Con video:

```powershell
python .\09_RA_Carnet\ra_carnet.py --dev .\ruta\video.mp4 --replace .\ruta\mi_foto.png
```

## Flujo de uso

1. Muestra el carnet de frente a la camara.
2. Pulsa `T` para capturar la plantilla.
3. Sobre la imagen rectificada que aparece, arrastra un rectangulo exactamente sobre la foto del carnet.
4. Confirma la seleccion con `ENTER` o `SPACE`.
5. A partir de ese momento, la sustitucion se hace automaticamente en tiempo real.

## Controles

- `T`: captura la plantilla inicial del carnet
- `R`: reinicia el suavizado temporal del seguimiento
- `Q` o `ESC`: salir

## Parametros utiles

- `--min-area-ratio`: area minima del carnet respecto al frame
- `--smooth`: suavizado temporal del cuadrilatero detectado
- `--hold`: cuantos frames se mantiene el ultimo carnet si se pierde la deteccion
- `--debug`: muestra el cuadrilatero del carnet y la subzona de la foto

Ejemplo con depuracion:

```powershell
python .\09_RA_Carnet\ra_carnet.py --dev 0 --replace .\ruta\mi_foto.png --debug
```

Ejemplo algo mas estable si la deteccion oscila:

```powershell
python .\09_RA_Carnet\ra_carnet.py --dev 0 --replace .\ruta\mi_foto.png --smooth 0.45 --hold 10
```

## Observaciones

- Funciona mejor si el carnet ocupa una zona razonable de la imagen, esta completo y se ve con buen contraste respecto al fondo.
- El metodo asume un carnet aproximadamente rectangular y con proporcion similar a una tarjeta estandar.
- La ventaja de capturar la plantilla desde la propia app es que la zona de la foto deja de ser aproximada y pasa a medirse sobre tu carnet real.
- Si la sustitucion se mantiene bien sobre el carnet pero vibra ligeramente, sube `--smooth` o `--hold`.
