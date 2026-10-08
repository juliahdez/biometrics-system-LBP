"""
Extractor de características LBP (Local Binary Patterns) y comparación.

Interfaz acordada:
    extraer(cara) -> template          (vector float32)
    comparar(t1, t2) -> distancia      (chi-cuadrado, 0 = idénticos)

Método (Ahonen et al., 2006):
  1. Para cada píxel se comparan sus P vecinos en un círculo de radio R:
     vecino >= centro -> 1, si no -> 0. Los P bits forman un código.
  2. Se usan solo los patrones "uniformes" (como mucho 2 cambios 0->1 / 1->0),
     que son los que aparecen en bordes, esquinas y manchas; el resto se agrupan
     en una sola casilla. Con P=8 salen 59 casillas.
  3. La cara se divide en una rejilla y se calcula un histograma por celda,
     para no perder DÓNDE está cada textura (ojos, boca...).
  4. El template es la concatenación de los histogramas normalizados.
"""
from __future__ import annotations

import numpy as np
from skimage.feature import local_binary_pattern

# Parámetros (ajustados con evaluar.py)
PUNTOS = 8          # P: vecinos
RADIO = 2           # R: radio del círculo
REJILLA = (8, 8)    # celdas (filas, columnas)
TAM_CARA = 128

N_CASILLAS = PUNTOS * (PUNTOS - 1) + 3          # 59 con P=8 ("nri_uniform")
TAM_TEMPLATE = REJILLA[0] * REJILLA[1] * N_CASILLAS


class ErrorTemplate(Exception):
    pass


def extraer(cara: np.ndarray) -> np.ndarray:
    """cara: 128x128 uint8 en gris (salida de preprocesar)."""
    if not isinstance(cara, np.ndarray) or cara.shape != (TAM_CARA, TAM_CARA):
        raise ErrorTemplate(f"La cara debe ser un array {TAM_CARA}x{TAM_CARA} en gris, "
                            f"no {getattr(cara, 'shape', type(cara).__name__)}")
    if cara.dtype != np.uint8:
        raise ErrorTemplate(f"La cara debe ser uint8, no {cara.dtype}")

    codigos = local_binary_pattern(cara, PUNTOS, RADIO, method="nri_uniform")

    filas, cols = REJILLA
    alto, ancho = TAM_CARA // filas, TAM_CARA // cols
    histos = []
    for i in range(filas):
        for j in range(cols):
            celda = codigos[i * alto:(i + 1) * alto, j * ancho:(j + 1) * ancho]
            h = np.bincount(celda.ravel().astype(np.int64), minlength=N_CASILLAS).astype(np.float64)
            histos.append(h / h.sum())           # cada celda suma 1
    return np.concatenate(histos).astype(np.float32)


def validar_template(t) -> np.ndarray:
    """Comprueba que un template tiene el formato correcto (clave para los métodos _t)."""
    try:
        t = np.asarray(t, dtype=np.float64)
    except (TypeError, ValueError):
        raise ErrorTemplate("El template no es numérico")
    if t.shape != (TAM_TEMPLATE,):
        raise ErrorTemplate(f"Tamaño de template incorrecto: {t.shape}, se esperaba ({TAM_TEMPLATE},)")
    if not np.isfinite(t).all():
        raise ErrorTemplate("El template contiene NaN o infinitos")
    if (t < 0).any() or (t > 1).any():
        raise ErrorTemplate("El template tiene valores fuera de [0, 1]")
    sumas = t.reshape(-1, N_CASILLAS).sum(axis=1)
    if not np.allclose(sumas, 1.0, atol=1e-3):
        raise ErrorTemplate("Los histogramas del template no están normalizados")
    return t


def comparar(t1, t2) -> float:
    """Distancia chi-cuadrado media por celda. 0 = idénticos, 2 = nada en común."""
    a, b = validar_template(t1), validar_template(t2)
    suma = a + b
    mascara = suma > 0
    chi2 = ((a - b) ** 2)[mascara] / suma[mascara]
    return float(chi2.sum() / (REJILLA[0] * REJILLA[1]))
