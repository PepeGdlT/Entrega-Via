# 06 - Rectificacion

Rectificacion de un plano para medir distancias reales a partir de referencias conocidas.

La idea es:

1. marcar en la imagen varios puntos de referencia del mismo plano
2. asociarlos con sus coordenadas reales
3. calcular la homografia `imagen -> plano`
4. rectificar para comprobar el resultado
5. medir distancias reales y mostrarlas sobre la imagen original

## Archivos

- `rectificacion.py`: calcula homografia, muestra la imagen rectificada y mide sobre la original
- `pick_refs.py`: genera un fichero de referencias haciendo click sobre la imagen
- `refs/a4_template.txt`: plantilla base para un A4

## Formato del fichero de referencias

```text
SCALE 3.0
REF p1 0 0 123.0 456.0
REF p2 210 0 512.0 430.0
REF p3 210 297 545.0 120.0
REF p4 0 297 100.0 140.0
MEASURE m1 250.0 300.0
MEASURE m2 420.0 310.0
```

Significado:

- `SCALE`: pixeles por unidad real para visualizar la rectificacion
- `REF label X Y x y`:
  - `X,Y` son coordenadas reales del plano
  - `x,y` son coordenadas de imagen
- `MEASURE label x y`:
  - punto opcional que ya quieres medir desde fichero

## Flujo recomendado para una imagen tuya

La opcion mas practica es usar una hoja A4 o una tarjeta.

### Opcion A4

- Medidas reales: `210 x 297 mm`
- Puntos reales recomendados:
  - `p1 = (0,0)`
  - `p2 = (210,0)`
  - `p3 = (210,297)`
  - `p4 = (0,297)`

Primero genera el fichero de referencias con clicks:

```powershell
python .\06_Rectificacion\pick_refs.py --image .\ruta\mi_foto.jpg --world "0,0;210,0;210,297;0,297" --labels "p1,p2,p3,p4" --scale 3.0 --output .\06_Rectificacion\refs\mi_a4.txt
```

Haz click en las 4 esquinas visibles de la hoja en ese mismo orden.

Luego lanza la rectificacion:

```powershell
python .\06_Rectificacion\rectificacion.py --image .\ruta\mi_foto.jpg --refs .\06_Rectificacion\refs\mi_a4.txt --units mm
```

## Uso de `rectificacion.py`

```powershell
python .\06_Rectificacion\rectificacion.py --image .\ruta\imagen.jpg --refs .\06_Rectificacion\refs\mis_refs.txt --units mm
```

Controles:

- click izquierdo dos veces: mide una distancia
- `C`: limpia la medicion interactiva
- `Q` o `ESC`: salir

El programa:

- dibuja las referencias sobre la imagen original
- muestra la imagen rectificada en otra ventana
- imprime en consola las coordenadas reales de los puntos medidos
- escribe la distancia sobre la imagen original

## Verificacion que conviene hacer

Para cumplir bien el ejercicio:

1. prueba primero con una referencia plana simple tomada por ti
2. mide una distancia conocida de verdad
3. comprueba el error

Ejemplos razonables:

- ancho de una hoja A4: `210 mm`
- lado corto de una tarjeta: `53.98 mm`
- lado largo de una tarjeta: `85.60 mm`
- distancia entre dos marcas hechas por ti con una regla

## Observaciones

- Las referencias deben estar todas en el mismo plano.
- Cuanto mejor repartidos esten los puntos de referencia, mejor sale la homografia.
- La medida sera peor lejos de la zona cubierta por esas referencias.
- Aunque se muestre la imagen rectificada para comprobar, la medicion final debe verse sobre la imagen original, que es justo lo que hace `rectificacion.py`.
