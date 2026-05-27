import numpy as np

import actividad


def run():
    # Solape nulo y parcial.
    assert actividad.intersection_ratio((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert actividad.intersection_ratio((0, 0, 20, 20), (10, 10, 25, 25)) > 0.0

    # Red ratio debe ser alto en un parche rojo.
    img_red = np.zeros((100, 100, 3), dtype=np.uint8)
    img_red[:] = (0, 0, 255)
    rr = actividad.red_ratio_hsv(img_red)
    assert rr > 0.90

    # Blur debe modificar el contenido del bbox.
    textured = np.zeros((80, 80, 3), dtype=np.uint8)
    patch = np.arange(40 * 40 * 3, dtype=np.uint8).reshape(40, 40, 3)
    textured[20:60, 20:60] = patch
    before = textured.copy()
    actividad.blur_box(textured, (20, 20, 60, 60))
    assert not np.array_equal(before, textured)

    print("test_actividad.py OK")


if __name__ == "__main__":
    run()
