# RA Sudoku

Ejercicio opcional: detectar un sudoku en vivo, leer sus numeros, resolverlo y proyectar la solucion sobre la imagen original.

Ejecucion desde la raiz del proyecto:

```bat
python 11_RA_Sudoku\ra_sudoku.py --dev 0
```

Tambien se puede probar con una imagen:

```bat
python 11_RA_Sudoku\ra_sudoku.py --dev .\umucv\images\sudoku.png
```

Controles:

- `R`: reinicia la lectura temporal.
- `S`: guarda una captura en `11_RA_Sudoku/captures`.
- `D`: activa/desactiva la ventana de depuracion OCR.
- `Q` o `ESC`: sale.

Funciona mejor con sudokus impresos, buena luz y el tablero ocupando una parte grande de la imagen.
