import sys
from pathlib import Path

# 1. Parche para encontrar la librería local del profesor
ROOT = Path(__file__).resolve().parents[1]
UMUCV_PKG = ROOT / "umucv" / "package"
if str(UMUCV_PKG) not in sys.path:
    sys.path.insert(0, str(UMUCV_PKG))

# 2. Ahora sí, importamos umucv y lo demás
from umucv.stream import autoStream
from umucv.util import Slider

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import time



# ---------------------------------------------------------------------------
# SUSTRACTOR DE FONDO
# ---------------------------------------------------------------------------
substractor = cv.createBackgroundSubtractorMOG2(
    history=500, varThreshold=40, detectShadows=True
)

# ---------------------------------------------------------------------------
# SLIDERS INTERACTIVOS  (Slider('nombre', 'ventana', default, min, max, paso))
# ---------------------------------------------------------------------------
# ┌─────────────┬──────────────────────────────────────────────────────────────┐
# │ Area min    │ Tamaño mínimo (px²) que debe tener un blob blanco en la      │
# │             │ máscara para considerarse un vehículo. Súbelo si detecta     │
# │             │ demasiado ruido; bájalo si no detecta vehículos lejanos.     │
# ├─────────────┼──────────────────────────────────────────────────────────────┤
# │ Umbral mask │ Valor de gris (0-254) a partir del cual un píxel de la       │
# │             │ máscara MOG2 se considera "en movimiento". MOG2 pinta las    │
# │             │ sombras en gris ~127 y el primer plano en blanco 255.        │
# │             │ Con 200 se eliminan sombras; bájalo a ~100 para incluirlas.  │
# ├─────────────┼──────────────────────────────────────────────────────────────┤
# │ Zona %      │ Semiancho de la banda central de conteo, en % del ancho del  │
# │             │ frame. Un vehículo solo cuenta cuando CRUZA completamente    │
# │             │ esta banda (de izq a der o viceversa). Si la pones muy       │
# │             │ pequeña puede haber dobles conteos; muy grande, puede        │
# │             │ confundir el sentido en cámaras con poco ángulo.             │
# ├─────────────┼──────────────────────────────────────────────────────────────┤
# │ ROI top %   │ Fila superior de la zona de análisis, en % del alto del      │
# │             │ frame. Súbelo para ignorar el cielo/señales en la parte      │
# │             │ superior de la imagen.                                       │
# ├─────────────┼──────────────────────────────────────────────────────────────┤
# │ ROI bot %   │ Fila inferior de la zona de análisis, en % del alto.         │
# │             │ Bájalo para ignorar el capó de la cámara o el bordillo       │
# │             │ en la parte inferior.                                        │
# └─────────────┴──────────────────────────────────────────────────────────────┘
cv.namedWindow('Trafico VIA')
# Defaults afinados segun los valores que mejor te funcionaron en la prueba visual.
sl_area    = Slider('Area min',   'Trafico VIA',   5,   0, 3000, 1)
sl_umbral  = Slider('Umbral mask','Trafico VIA',   40,   0,  254, 1)
sl_zona    = Slider('Zona %',     'Trafico VIA',   8,   2,   40, 1)
sl_roi_top = Slider('ROI top %',  'Trafico VIA',  43,   0,   90, 1)
sl_roi_bot = Slider('ROI bot %',  'Trafico VIA',  63,   0,  100, 1)

# ---------------------------------------------------------------------------
# CONTADORES Y ESTADO
# ---------------------------------------------------------------------------
van_derecha   = 0
van_izquierda = 0
frame_count   = 0
fps_estimado  = 30

historial = []  # (t_seg, der_acum, izq_acum, flujo_der, flujo_izq)

VENTANA_FLUJO = 60
eventos_der = deque()
eventos_izq = deque()

# ---------------------------------------------------------------------------
# TRACKER SIMPLE POR ID
# ---------------------------------------------------------------------------
siguiente_id = 0
vehiculos    = {}
MAX_PERDIDO  = 8
MAX_DIST     = 120

def emparejar_vehiculos(vehiculos, detecciones):
    global siguiente_id
    emparejados    = {}
    no_emparejados = list(detecciones)

    for vid, v in list(vehiculos.items()):
        px, py = v['pos']
        mejor_d, mejor_idx = MAX_DIST, -1
        for i, (cx, cy) in enumerate(no_emparejados):
            d = np.hypot(cx - px, cy - py)
            if d < mejor_d:
                mejor_d, mejor_idx = d, i
        if mejor_idx != -1:
            emparejados[vid] = no_emparejados.pop(mejor_idx)

    for vid in list(vehiculos.keys()):
        if vid in emparejados:
            vehiculos[vid]['pos']     = emparejados[vid]
            vehiculos[vid]['perdido'] = 0
        else:
            vehiculos[vid]['perdido'] += 1

    for vid in [v for v, d in vehiculos.items() if d['perdido'] > MAX_PERDIDO]:
        del vehiculos[vid]

    nuevos = []
    for cx, cy in no_emparejados:
        nuevos.append((siguiente_id, cx, cy))
        siguiente_id += 1
    return nuevos

# ---------------------------------------------------------------------------
# BUCLE PRINCIPAL
# ---------------------------------------------------------------------------
t_inicio       = time.time()
t_ultimo_frame = time.time()

for key, frame in autoStream():

    ahora = time.time()
    fps_estimado   = 0.9 * fps_estimado + 0.1 * (1.0 / max(ahora - t_ultimo_frame, 1e-6))
    t_ultimo_frame = ahora
    frame_count   += 1
    t_seg          = ahora - t_inicio

    h, w = frame.shape[:2]

    # --- Parámetros desde sliders ---
    area_min    = sl_area.value
    umbral_mask = sl_umbral.value
    semiancho_z = int(w * sl_zona.value    / 100)
    roi_top     = int(h * sl_roi_top.value / 100)
    roi_bot     = int(h * sl_roi_bot.value / 100)
    if roi_bot <= roi_top + 10:
        roi_bot = roi_top + 10

    linea_conteo = w // 2
    zona_izq     = linea_conteo - semiancho_z
    zona_der     = linea_conteo + semiancho_z

    # --- Sustracción de fondo solo en el ROI ---
    frame_roi   = frame[roi_top:roi_bot, :]
    mascara_roi = substractor.apply(frame_roi)

    # Umbral configurable (slider) en vez de fijo a 200
    _, mascara_roi = cv.threshold(mascara_roi, umbral_mask, 255, cv.THRESH_BINARY)

    k_open  = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,  5))
    k_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    mascara_roi = cv.morphologyEx(mascara_roi, cv.MORPH_OPEN,  k_open)
    mascara_roi = cv.morphologyEx(mascara_roi, cv.MORPH_CLOSE, k_close)

    mascara_full = np.zeros((h, w), dtype=np.uint8)
    mascara_full[roi_top:roi_bot, :] = mascara_roi

    # --- Detección de contornos ---
    contornos, _ = cv.findContours(mascara_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detecciones = []
    for c in contornos:
        if cv.contourArea(c) > area_min:
            x, y, ancho, alto = cv.boundingRect(c)
            aspecto = ancho / max(alto, 1)
            if 0.3 < aspecto < 8.0:
                cx = x + ancho // 2
                cy = y + alto  // 2 + roi_top
                detecciones.append((cx, cy))
                cv.rectangle(frame,
                             (x, y + roi_top),
                             (x + ancho, y + alto + roi_top),
                             (0, 200, 0), 2)
                cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # --- Tracking y detección de cruces ---
    nuevos = emparejar_vehiculos(vehiculos, detecciones)

    for vid, cx, cy in nuevos:
        lado_ini = 'izq' if cx < linea_conteo else 'der'
        vehiculos[vid] = {'pos': (cx, cy), 'lado': lado_ini, 'perdido': 0}

    for vid, v in vehiculos.items():
        cx, cy = v['pos']
        if not (roi_top <= cy <= roi_bot):
            continue
        if v['lado'] == 'izq' and cx > zona_der:
            van_derecha += 1
            eventos_der.append(t_seg)
            v['lado'] = 'der'
        elif v['lado'] == 'der' and cx < zona_izq:
            van_izquierda += 1
            eventos_izq.append(t_seg)
            v['lado'] = 'izq'

    while eventos_der and t_seg - eventos_der[0] > VENTANA_FLUJO:
        eventos_der.popleft()
    while eventos_izq and t_seg - eventos_izq[0] > VENTANA_FLUJO:
        eventos_izq.popleft()

    flujo_der = len(eventos_der) / VENTANA_FLUJO * 60
    flujo_izq = len(eventos_izq) / VENTANA_FLUJO * 60

    if frame_count % max(1, int(fps_estimado)) == 0:
        historial.append((t_seg, van_derecha, van_izquierda, flujo_der, flujo_izq))

    # --- HUD ---
    cv.rectangle(frame, (0, roi_top), (w - 1, roi_bot), (0, 200, 200), 1)

    overlay = frame.copy()
    cv.rectangle(overlay, (zona_izq, roi_top), (zona_der, roi_bot), (255, 100, 0), -1)
    cv.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    cv.line(frame, (linea_conteo, roi_top), (linea_conteo, roi_bot), (0, 0, 255), 2)

    cv.putText(frame, f"<-- IZQ: {van_izquierda}",
               (20, 45), cv.FONT_HERSHEY_SIMPLEX, 1.1, (255, 200, 0), 3)
    cv.putText(frame, f"DER: {van_derecha} -->",
               (w - 300, 45), cv.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 255), 3)

    cv.putText(frame, f"Flujo izq: {flujo_izq:.1f} veh/min",
               (20, 85), cv.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
    cv.putText(frame, f"Flujo der: {flujo_der:.1f} veh/min",
               (20, 115), cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # Guía rápida de sliders en esquina inferior derecha
    guia = [
        "SLIDERS:",
        f"Area min {area_min}px2: tamano minimo blob",
        f"Umbral mask {umbral_mask}: 200=sin sombras 100=con sombras",
        f"Zona % {sl_zona.value}%: ancho banda conteo central",
        f"ROI top/bot: zona de analisis (linea cyan)",
    ]
    for i, linea in enumerate(guia):
        cv.putText(frame, linea,
                   (10, h - 15 - (len(guia) - 1 - i) * 18),
                   cv.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1)

    mins, secs = divmod(int(t_seg), 60)
    cv.putText(frame, f"T:{mins:02d}:{secs:02d} FPS:{fps_estimado:.0f}  [R]=reset [ESC]=salir+graficas",
               (w - 430, h - 8), cv.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    cv.imshow('Trafico VIA', frame)
    cv.imshow('Mascara', mascara_full)

    # --- Teclado ---
    if key == 27:          # ESC → salir
        break
    elif key == ord('r'):  # R   → resetear contadores
        van_derecha = van_izquierda = frame_count = 0
        historial.clear()
        eventos_der.clear()
        eventos_izq.clear()
        t_inicio = time.time()

cv.destroyAllWindows()

# ---------------------------------------------------------------------------
# GRÁFICAS FINALES
# ---------------------------------------------------------------------------
if len(historial) < 2:
    print("No hay suficientes datos para generar gráficas.")
else:
    tiempos  = np.array([r[0] for r in historial])
    acum_der = np.array([r[1] for r in historial])
    acum_izq = np.array([r[2] for r in historial])
    fl_der   = np.array([r[3] for r in historial])
    fl_izq   = np.array([r[4] for r in historial])
    t_min    = tiempos / 60.0   # eje X en minutos

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Análisis de Tráfico — Vehículos por sentido',
                 fontsize=15, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Acumulado ambos sentidos (fila completa superior)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t_min, acum_der, color='#2196F3', linewidth=2, label='→ Derecha')
    ax1.plot(t_min, acum_izq, color='#FF5722', linewidth=2, label='← Izquierda')
    ax1.fill_between(t_min, acum_der, alpha=0.15, color='#2196F3')
    ax1.fill_between(t_min, acum_izq, alpha=0.15, color='#FF5722')
    ax1.set_title('Vehículos acumulados por sentido')
    ax1.set_xlabel('Tiempo (min)')
    ax1.set_ylabel('Vehículos totales')
    ax1.legend(); ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Flujo instantáneo → derecha
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t_min, fl_der, color='#2196F3', linewidth=1.5)
    ax2.fill_between(t_min, fl_der, alpha=0.2, color='#2196F3')
    idx = int(np.argmax(fl_der))
    ax2.axvline(float(t_min[idx]), color='red', linestyle='--', linewidth=1,
                label=f'Punta: {fl_der[idx]:.1f} v/min @ {t_min[idx]:.1f} min')
    ax2.set_title('Flujo instantáneo → Derecha')
    ax2.set_xlabel('Tiempo (min)'); ax2.set_ylabel('Vehículos/min')
    ax2.legend(fontsize=8); ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Flujo instantáneo ← izquierda
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t_min, fl_izq, color='#FF5722', linewidth=1.5)
    ax3.fill_between(t_min, fl_izq, alpha=0.2, color='#FF5722')
    idx = int(np.argmax(fl_izq))
    ax3.axvline(float(t_min[idx]), color='red', linestyle='--', linewidth=1,
                label=f'Punta: {fl_izq[idx]:.1f} v/min @ {t_min[idx]:.1f} min')
    ax3.set_title('Flujo instantáneo ← Izquierda')
    ax3.set_xlabel('Tiempo (min)'); ax3.set_ylabel('Vehículos/min')
    ax3.legend(fontsize=8); ax3.grid(True, linestyle='--', alpha=0.5)

    # 4. Flujo total con umbral de hora punta automático (percentil 75)
    ax4 = fig.add_subplot(gs[2, 0])
    flujo_total  = fl_der + fl_izq
    umbral_punta = float(np.percentile(flujo_total, 75)) if len(flujo_total) > 4 else 0.0
    ax4.plot(t_min, flujo_total, color='#4CAF50', linewidth=1.5)
    ax4.fill_between(t_min, flujo_total, alpha=0.2, color='#4CAF50')
    ax4.axhline(umbral_punta, color='orange', linestyle=':', linewidth=1.5,
                label=f'Umbral hora punta (p75): {umbral_punta:.1f} v/min')
    ax4.fill_between(t_min, umbral_punta, flujo_total,
                     where=(flujo_total >= umbral_punta),
                     alpha=0.35, color='orange', label='Hora punta')
    ax4.set_title('Flujo total ambos sentidos')
    ax4.set_xlabel('Tiempo (min)'); ax4.set_ylabel('Vehículos/min')
    ax4.legend(fontsize=8); ax4.grid(True, linestyle='--', alpha=0.5)

    # 5. Balance acumulado Der − Izq
    ax5 = fig.add_subplot(gs[2, 1])
    balance = acum_der - acum_izq
    colores  = ['#2196F3' if b >= 0 else '#FF5722' for b in balance]
    ancho_barra = float(t_min[1] - t_min[0]) if len(t_min) > 1 else 1.0
    ax5.bar(t_min, balance, width=ancho_barra, color=colores, alpha=0.7)
    ax5.axhline(0, color='black', linewidth=0.8)
    ax5.set_title('Balance acumulado (Der − Izq)')
    ax5.set_xlabel('Tiempo (min)'); ax5.set_ylabel('Diferencia vehículos')
    ax5.grid(True, linestyle='--', alpha=0.5, axis='y')

    out_path = Path(__file__).with_name('trafico_analisis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Gráfica guardada en {out_path}")
    plt.show()
