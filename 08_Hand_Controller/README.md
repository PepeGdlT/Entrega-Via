# 08 - Hand Controller

Controlador sin contacto basado en MediaPipe Hands para manejar un objeto virtual 2D.

## Grados de libertad usados

- `x`: posicion horizontal de la mano
- `y`: posicion vertical de la mano
- `distancia` aproximada a camara:
  - se estima con el ancho aparente de la palma
- `angulo` de orientacion:
  - se estima con el eje muneca -> dedo medio

## Efecto sobre el objeto virtual

- mover la mano: traslada el objeto
- acercar la mano: hace el objeto mas grande
- alejar la mano: hace el objeto mas pequeno
- girar la mano: rota el objeto

## Ejecucion

Con webcam:

```powershell
python .\08_Hand_Controller\hand_controller.py --dev 0
```

Con un suavizado algo mayor:

```powershell
python .\08_Hand_Controller\hand_controller.py --dev 0 --smooth 0.35
```

## Controles

- `B`: fija el tamano actual de la mano como referencia para `scale=1`
- `R`: reinicia el suavizado
- `Q` o `ESC`: salir

## Observaciones

- El modo espejo esta activado por defecto para que el control sea natural.
- La distancia a camara es aproximada, basada en tamano aparente, no en profundidad real metrizada.
- Si la escala sale rara, acerca la mano a una posicion comoda y pulsa `B`.
