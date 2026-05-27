# 01 - Calibracion

Script principal: `calibracion.py`

## 1) Capturar/calibrar con chessboard

Usa un patron interior `9x6` (por defecto) y pulsa `c` para guardar cada vista valida.
Cuando tengas suficientes vistas inclinadas (12+), pulsa `ENTER` para calibrar y guardar `calib.txt`.

```powershell
python .\01_Calibracion\calibracion.py --mode calibrate --dev "glob:.\umucv\code\calibrate\mylogitech\*.png" --pattern 9x6 --square-size 1.0 --min-views 12
```

## 2) Medidor con rejilla metrica

Carga `calib.txt` y superpone una rejilla en un plano a distancia `Z`.

```powershell
python .\01_Calibracion\calibracion.py --mode overlay --dev 0 --calib .\01_Calibracion\calib.txt --height0 0.8 --z0 2.0
```

Tambien puedes usar video:

```powershell
python .\01_Calibracion\calibracion.py --mode overlay --dev .\03_Actividad\videos\TownCentre.mp4 --calib .\01_Calibracion\calib.txt
```

## Sliders (ventana `medidor`)

- `fov`: ajusta el FOV horizontal efectivo.
- `Z`: distancia del plano de medida en metros.
- `A`: altura de camara en metros.
- `X`: desplazamiento lateral de la rejilla (metros).

La linea gris horizontal marca el centro optico de la camara.
Con camara horizontal, deberia coincidir aprox. con la linea de altura `A` en la rejilla.

---


[calibrate] Pulsa 'c' para capturar una pose valida del chessboard.
[calibrate] Pulsa ENTER para calibrar y guardar. ESC/q para salir.
1280x720 30.0fps
[calibrate] Captura 1 guardada
[calibrate] Captura 2 guardada
[calibrate] Captura 3 guardada
[calibrate] Captura 4 guardada
[calibrate] Captura 5 guardada
[calibrate] Captura 6 guardada
[calibrate] Captura 7 guardada
[calibrate] Captura 8 guardada
[calibrate] Captura 9 guardada
[calibrate] Captura 10 guardada
[calibrate] Captura 11 guardada
[calibrate] Captura 12 guardada
[calibrate] Captura 13 guardada
[calibrate] Captura 14 guardada
[calibrate] Captura 15 guardada
[calibrate] Captura 16 guardada
[calibrate] Captura 17 guardada
[calibrate] Captura 18 guardada
[calibrate] Captura 19 guardada
[calibrate] Captura 20 guardada
[calibrate] RMS: 0.1875
[calibrate] K:
[[  872.519     0.000   638.904]
 [    0.000   872.818   357.877]
 [    0.000     0.000     1.000]]
[calibrate] D: [ 1.416040e-01  1.117930e-01 -6.410000e-04  8.860000e-04 -1.992824e+00]
[calibrate] FOV: h=72.52 deg, v=44.83 deg
[calibrate] Guardado en: .\01_Calibracion\calib.txt

[overlay] chessman elite mide 41cm
1280x720 30.0fps
[overlay] Resolucion: 1280x720
[overlay] FOV calibrado: h=72.52 deg, v=44.83 deg