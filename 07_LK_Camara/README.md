# 07 - LK Camara

Ampliacion de `umucv/code/LK/lk_track.py` para estimar:

- direccion del movimiento de la camara:
  - `UP`, `DOWN`, `LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`
- velocidad angular aproximada:
  - `yaw` y `pitch` en `deg/s`

## Idea

Se detectan esquinas y se siguen entre frames con Lucas-Kanade.

Con los desplazamientos de los puntos:

- media horizontal `dx` y vertical `dy`:
  - sirven para clasificar `LEFT/RIGHT/UP/DOWN`
- cambio radial medio respecto al centro `dr`:
  - sirve para clasificar `FORWARD/BACKWARD`

La velocidad angular aproximada se calcula convirtiendo `dx` y `dy` a angulo usando la focal en pixeles.

## Ejecucion

Con webcam:

```powershell
python .\07_LK_Camara\lk_camera_motion.py --dev 0
```

Con calibracion de la practica 01:

```powershell
python .\07_LK_Camara\lk_camera_motion.py --dev 0 --calib .\01_Calibracion\calib.txt
```

Con video:

```powershell
python .\07_LK_Camara\lk_camera_motion.py --dev .\ruta\video.mp4
```

## Controles

- `C`: reinicia las trayectorias
- `Q` o `ESC`: salir

## Parametros utiles

- `--dx-thr`: umbral horizontal medio para activar `LEFT/RIGHT`
- `--dy-thr`: umbral vertical medio para activar `UP/DOWN`
- `--dr-thr`: umbral radial medio para activar `FORWARD/BACKWARD`
- `--smooth`: suavizado temporal de la estimacion
- `--hfov`: FOV horizontal por defecto si no usas calibracion

Ejemplo mas estable:

```powershell
python .\07_LK_Camara\lk_camera_motion.py --dev 0 --dx-thr 2.0 --dy-thr 2.0 --dr-thr 1.6 --smooth 10
```

## Observaciones

- La direccion estimada es aproximada.
- Funciona mejor en escenas con textura y poco movimiento de objetos independientes.
- `FORWARD/BACKWARD` se detecta por expansion o contraccion radial de los tracks.
- La velocidad angular es orientativa, no una medida calibrada de alta precision.
