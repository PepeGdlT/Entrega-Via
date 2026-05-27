# 10 - Algoritmo Propio

Implementacion propia del detector de esquinas de Harris y comparacion con OpenCV.

## Algoritmo implementado

Version propia de Harris:

1. suavizado gaussiano
2. gradientes `Ix`, `Iy`
3. productos `Ix^2`, `Iy^2`, `IxIy`
4. suavizado local de esos productos
5. respuesta:

```text
R = det(M) - k * trace(M)^2
```

6. normalizacion
7. supresion de no maximos
8. umbral relativo y seleccion de picos

Se compara con `cv.cornerHarris`.

## Ejecucion

Con webcam:

```powershell
python .\10_Algoritmo_Propio\harris_compare.py --dev 0
```

Con video:

```powershell
python .\10_Algoritmo_Propio\harris_compare.py --dev .\ruta\video.mp4
```

## Que muestra

- arriba izquierda: esquinas del Harris propio
- arriba derecha: esquinas de `cv.cornerHarris`
- abajo: mapas de respuesta normalizados de ambos
- tiempos de calculo
- numero de esquinas detectadas
- porcentaje aproximado de solape entre ambos detectores

## Parametros utiles

- `--sigma-grad`: suavizado previo al gradiente
- `--sigma-window`: tamano efectivo de la ventana local
- `--kappa`: constante de Harris
- `--quality`: umbral relativo tras normalizacion
- `--max-corners`: maximo de esquinas a mostrar
- `--nms-size`: tamano de la supresion de no maximos
- `--match-radius`: tolerancia para medir coincidencia entre detectores

Ejemplo mas conservador:

```powershell
python .\10_Algoritmo_Propio\harris_compare.py --dev 0 --quality 0.20 --max-corners 150
```

## Interpretacion

- Si el algoritmo propio esta bien, los mapas de respuesta deben parecerse.
- Las esquinas no tienen por que coincidir pixel a pixel, por eso se mide el solape con un radio de tolerancia.
- OpenCV normalmente sera mas rapido o mas estable, pero la implementacion propia debe reproducir el comportamiento general.
