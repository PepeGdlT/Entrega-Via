# RA Mouse 3D

Ejercicio opcional: efecto de realidad aumentada 3D donde el usuario marca posiciones con el raton sobre un chessboard real y los cubos virtuales se desplazan suavemente hacia esos puntos.

El chessboard de calibracion se usa como plano de referencia. El programa detecta sus esquinas, estima la pose con `solvePnP` y proyecta cubos 3D con la calibracion de la camara.

Ejecucion desde la raiz del proyecto:

```bat
python 12_RA_Mouse\ra_mouse.py --dev 0 --calib .\01_Calibracion\calib.txt --pattern 9x6
```

Controles:

- `N`: modo crear. El siguiente click izquierdo crea un cubo 3D en el tablero.
- `E`: modo seleccionar. El siguiente click izquierdo selecciona el cubo cercano.
- `M`: modo mover. El siguiente click izquierdo mueve el cubo seleccionado a esa posicion del tablero.
- Click derecho: elimina el cubo cercano.
- `C`: limpia todos los objetos.
- `S`: guarda una captura en `12_RA_Mouse/captures`.
- `Q` o `ESC`: sale.
