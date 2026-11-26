"""
Preprocesado de la cara antes de extraer el LBP.

    preprocesar(foto, caja, ojo_izq, ojo_der) -> cara 128x128 uint8   (con detector)
    preprocesar_sin_detector(foto)            -> cara 128x128 uint8   (métodos _sd)

Pasos:
  1. Alineación: transformación de semejanza (giro + escala + traslación) que lleva
     los dos ojos a posiciones fijas. Así cada celda de la rejilla LBP cae siempre
     sobre la misma zona de la cara (ceja, ojo, nariz...).
  2. Escala de grises.
  3. Normalización de iluminación de Tan y Triggs (2010): corrección gamma, filtro
     diferencia de gaussianas y ecualización de contraste. Es la normalización
     estándar para LBP con iluminación no controlada; con nuestras fotos de prueba
     bajó el EER del 30 % al 18 % frente a CLAHE.
"""
from __future__ import annotations

import cv2
import numpy as np

from detector import ImagenInvalidaError, cargar_imagen

TAM_CARA = 128
# Posición de los ojos en la cara final (fracción del lado). Recorte ajustado a la
# cara: deja fuera casi todo el pelo y el fondo, que no identifican a la persona.
OJO_IZQ_DESTINO = (0.27, 0.32)
OJO_DER_DESTINO = (0.73, 0.32)

# Posición media de los ojos DENTRO de la caja de YuNet (medida sobre 59 fotos
# reales). Se usa en los métodos _sd, donde no hay detector que dé los ojos: así
# una cara recortada por la caja se alinea igual que en la vía normal.
OJOS_EN_CAJA = ((0.261, 0.385), (0.726, 0.385))

DESV_MIN = 2.0   # una "cara" con menos desviación típica es una imagen lisa


class ErrorPreprocesado(ImagenInvalidaError):
    pass


# --------------------------------------------------------------------------- #
def _matriz_semejanza(ojo_izq, ojo_der, tam: int) -> np.ndarray:
    src = np.array([ojo_izq, ojo_der], dtype=np.float64)
    dst = np.array([OJO_IZQ_DESTINO, OJO_DER_DESTINO], dtype=np.float64) * tam
    vs, vd = src[1] - src[0], dst[1] - dst[0]
    escala = np.linalg.norm(vd) / np.linalg.norm(vs)
    angulo = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])
    c, s = escala * np.cos(angulo), escala * np.sin(angulo)
    R = np.array([[c, -s], [s, c]])
    t = dst[0] - R @ src[0]
    return np.hstack([R, t[:, None]])


def _ojos_validos(caja, ojo_izq, ojo_der) -> bool:
    """Los ojos tienen sentido respecto a la caja (si no, se alinea con la caja)."""
    x, y, w, h = caja
    dx, dy = ojo_der[0] - ojo_izq[0], ojo_der[1] - ojo_izq[1]
    distancia = float(np.hypot(dx, dy))
    dentro = all(x - 0.1 * w <= px <= x + 1.1 * w and y - 0.1 * h <= py <= y + 1.1 * h
                 for px, py in (ojo_izq, ojo_der))
    return dentro and 0.2 * w <= distancia <= 0.9 * w and dx > 0


def tan_triggs(cara: np.ndarray, gamma=0.2, sigma0=1.0, sigma1=2.0, alfa=0.1, tau=10.0) -> np.ndarray:
    x = np.power(cara.astype(np.float64) + 1.0, gamma)
    x = cv2.GaussianBlur(x, (0, 0), sigma0) - cv2.GaussianBlur(x, (0, 0), sigma1)
    eps = 1e-8
    x = x / (np.mean(np.abs(x) ** alfa) ** (1 / alfa) + eps)
    x = x / (np.mean(np.minimum(np.abs(x), tau) ** alfa) ** (1 / alfa) + eps)
    x = tau * np.tanh(x / tau)
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _normalizar(cara: np.ndarray) -> np.ndarray:
    if float(cara.std()) < DESV_MIN:
        raise ErrorPreprocesado("La cara es una imagen lisa (sin contenido)")
    return tan_triggs(cara)


# --------------------------------------------------------------------------- #
def preprocesar(foto, caja, ojo_izq, ojo_der, tam: int = TAM_CARA) -> np.ndarray:
    img = cargar_imagen(foto)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x, y, w, h = map(float, caja)
    if w <= 0 or h <= 0:
        raise ErrorPreprocesado(f"Caja no válida: {caja}")

    if not _ojos_validos(caja, ojo_izq, ojo_der):
        # Ojos poco fiables: se colocan en su posición media dentro de la caja
        ojo_izq = (x + OJOS_EN_CAJA[0][0] * w, y + OJOS_EN_CAJA[0][1] * h)
        ojo_der = (x + OJOS_EN_CAJA[1][0] * w, y + OJOS_EN_CAJA[1][1] * h)

    M = _matriz_semejanza(ojo_izq, ojo_der, tam)
    cara = cv2.warpAffine(gris, M, (tam, tam), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return _normalizar(cara)


def preprocesar_sin_detector(foto, tam: int = TAM_CARA) -> np.ndarray:
    """La foto entera se trata como si fuera el recorte de la caja del detector."""
    img = cargar_imagen(foto)
    h, w = img.shape[:2]
    return preprocesar(img, (0, 0, w, h),
                       (OJOS_EN_CAJA[0][0] * w, OJOS_EN_CAJA[0][1] * h),
                       (OJOS_EN_CAJA[1][0] * w, OJOS_EN_CAJA[1][1] * h), tam)
