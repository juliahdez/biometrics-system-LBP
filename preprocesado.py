"""
Preprocesado de la cara antes de extraer el LBP.

Interfaz acordada:
    preprocesar(foto, caja, ojo_izq, ojo_der) -> cara 128x128 en gris (uint8)

Pasos:
  1. Alineación: transformación de semejanza (giro + escala + traslación) que
     lleva los dos ojos a posiciones fijas. Así todas las caras quedan con los
     ojos horizontales, a la misma distancia y en el mismo sitio, y cada celda
     de la rejilla LBP cae siempre sobre la misma zona de la cara.
  2. Escala de grises.
  3. Normalización de iluminación con CLAHE (ecualización adaptativa).
"""
from __future__ import annotations

import cv2
import numpy as np

from detector import cargar_imagen

TAM_CARA = 128
# Posición de los ojos en la cara final (fracción del lado)
OJO_IZQ_DESTINO = (0.30, 0.36)
OJO_DER_DESTINO = (0.70, 0.36)

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


class ErrorPreprocesado(Exception):
    pass


def _ojos_validos(caja, ojo_izq, ojo_der) -> bool:
    """Comprueba que los ojos tienen sentido respecto a la caja."""
    x, y, w, h = caja
    dx, dy = ojo_der[0] - ojo_izq[0], ojo_der[1] - ojo_izq[1]
    distancia = np.hypot(dx, dy)
    dentro = all(x - 0.1 * w <= px <= x + 1.1 * w and y - 0.1 * h <= py <= y + 1.1 * h
                 for px, py in (ojo_izq, ojo_der))
    return dentro and 0.2 * w <= distancia <= 0.9 * w and dx > 0


def preprocesar(foto, caja, ojo_izq, ojo_der, tam: int = TAM_CARA) -> np.ndarray:
    img = cargar_imagen(foto)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x, y, w, h = map(float, caja)
    if w <= 0 or h <= 0:
        raise ErrorPreprocesado(f"Caja no válida: {caja}")

    if _ojos_validos(caja, ojo_izq, ojo_der):
        # Semejanza que manda ojo_izq -> destino_izq y ojo_der -> destino_der
        src = np.array([ojo_izq, ojo_der], dtype=np.float64)
        dst = np.array([OJO_IZQ_DESTINO, OJO_DER_DESTINO], dtype=np.float64) * tam
        vs, vd = src[1] - src[0], dst[1] - dst[0]
        escala = np.linalg.norm(vd) / np.linalg.norm(vs)
        angulo = np.arctan2(vd[1], vd[0]) - np.arctan2(vs[1], vs[0])
        c, s = escala * np.cos(angulo), escala * np.sin(angulo)
        R = np.array([[c, -s], [s, c]])
        t = dst[0] - R @ src[0]
        M = np.hstack([R, t[:, None]])
    else:
        # Ojos poco fiables: se usa solo la caja (recorte cuadrado centrado)
        lado = max(w, h)
        cx, cy = x + w / 2, y + h / 2
        escala = tam / lado
        M = np.array([[escala, 0, tam / 2 - escala * cx],
                      [0, escala, tam / 2 - escala * cy]])

    cara = cv2.warpAffine(gris, M, (tam, tam), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return _clahe.apply(cara)
