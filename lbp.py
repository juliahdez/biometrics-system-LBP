"""
Extractor de características LBP (Local Binary Patterns) y comparación.

    extraer(cara) -> template            vector float32 de TAM_TEMPLATE valores
    comparar(t1, t2) -> distancia        chi-cuadrado (0 = idénticos, 2 = nada en común)
    validar_template(t) -> np.ndarray    rechaza templates con formato incorrecto

Método (Ahonen, Hadid y Pietikäinen, 2006):
  1. Cada píxel se compara con P=8 vecinos en un círculo de radio R=2:
     vecino >= centro -> 1, si no -> 0. Los 8 bits forman un código de textura.
  2. Solo se distinguen los patrones "uniformes" (como mucho 2 cambios 0<->1): bordes,
     esquinas, manchas... El resto va a una casilla común. Con P=8 salen 59 casillas.
  3. La cara se divide en una rejilla de 10x10 celdas y se hace un histograma por
     celda, para conservar DÓNDE está cada textura.
  4. Template simétrico: se promedian los histogramas de la cara y de su imagen en
     espejo. Así el template es menos sensible a giros leves y a la luz lateral.
  5. Template = concatenación de los 100 histogramas normalizados (100 x 59 = 5900).
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

PUNTOS = 8
RADIO = 2
REJILLA = 10
TAM_CARA = 128
SIMETRICO = True

N_CASILLAS = PUNTOS * (PUNTOS - 1) + 3          # 59
N_CELDAS = REJILLA * REJILLA                    # 100
TAM_TEMPLATE = N_CELDAS * N_CASILLAS            # 5900
_BORDES = np.linspace(0, TAM_CARA, REJILLA + 1).round().astype(int)


class ErrorTemplate(ValueError):
    pass


def _histogramas(cara: np.ndarray) -> np.ndarray:
    codigos = local_binary_pattern(cara, PUNTOS, RADIO, method="nri_uniform").astype(np.int64)
    histos = np.empty((N_CELDAS, N_CASILLAS))
    k = 0
    for i in range(REJILLA):
        for j in range(REJILLA):
            celda = codigos[_BORDES[i]:_BORDES[i + 1], _BORDES[j]:_BORDES[j + 1]]
            h = np.bincount(celda.ravel(), minlength=N_CASILLAS).astype(np.float64)
            histos[k] = h / h.sum()
            k += 1
    return histos


def extraer(cara: np.ndarray) -> np.ndarray:
    """cara: 128x128 uint8 en gris (salida del preprocesado)."""
    if not isinstance(cara, np.ndarray) or cara.shape != (TAM_CARA, TAM_CARA):
        raise ErrorTemplate(f"La cara debe ser un array {TAM_CARA}x{TAM_CARA} en gris, "
                            f"no {getattr(cara, 'shape', type(cara).__name__)}")
    if cara.dtype != np.uint8:
        raise ErrorTemplate(f"La cara debe ser uint8, no {cara.dtype}")
    h = _histogramas(cara)
    if SIMETRICO:
        h = (h + _histogramas(cv2.flip(cara, 1))) / 2
    return h.ravel().astype(np.float32)


def validar_template(t) -> np.ndarray:
    """Comprueba que un template tiene exactamente el formato que produce extraer().

    Es la defensa de los métodos _t: un atacante no puede meter vectores de otro
    tamaño, con NaN/infinitos, negativos o histogramas que no sumen 1.
    """
    if isinstance(t, (str, bytes)) or t is None:
        raise ErrorTemplate("El template debe ser un vector numérico")
    try:
        t = np.asarray(t, dtype=np.float64)
    except (TypeError, ValueError):
        raise ErrorTemplate("El template no es numérico") from None
    if t.shape != (TAM_TEMPLATE,):
        raise ErrorTemplate(f"Tamaño de template incorrecto: {t.shape}, se esperaba ({TAM_TEMPLATE},)")
    if not np.isfinite(t).all():
        raise ErrorTemplate("El template contiene NaN o infinitos")
    if (t < 0).any() or (t > 1).any():
        raise ErrorTemplate("El template tiene valores fuera de [0, 1]")
    sumas = t.reshape(N_CELDAS, N_CASILLAS).sum(axis=1)
    if not np.allclose(sumas, 1.0, atol=1e-3):
        raise ErrorTemplate("Los histogramas del template no están normalizados (cada celda debe sumar 1)")
    return t


def distancias(t, matriz: np.ndarray) -> np.ndarray:
    """Distancia chi-cuadrado de un template contra una matriz de templates (N x TAM)."""
    t = validar_template(t)
    m = np.asarray(matriz, dtype=np.float64).reshape(-1, TAM_TEMPLATE)
    suma = t + m
    dif2 = (t - m) ** 2
    chi2 = np.divide(dif2, suma, out=np.zeros_like(dif2), where=suma > 0)
    return chi2.sum(axis=1) / N_CELDAS


def comparar(t1, t2) -> float:
    """Distancia chi-cuadrado media por celda: 0 = idénticos ... 2 = nada en común."""
    return float(distancias(t1, validar_template(t2)[None, :])[0])
