"""
Cadena completa: foto -> detector -> preprocesado -> LBP -> template.

    obtener_template(foto)              con detector (para inscribir / verificar / identificar)
    obtener_template_sd(foto)           sin detector (para los métodos _sd)
"""
from __future__ import annotations

import cv2

from detector import cargar_imagen, detectar
from lbp import comparar, extraer  # noqa: F401  (se reexporta comparar)
from preprocesado import TAM_CARA, preprocesar, _clahe


def obtener_template(foto):
    img = cargar_imagen(foto)
    caja, ojo_izq, ojo_der = detectar(img)
    return extraer(preprocesar(img, caja, ojo_izq, ojo_der))


def obtener_template_sd(foto):
    """Sin detector: se asume que la foto ya es la cara recortada."""
    img = cargar_imagen(foto)
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cara = cv2.resize(gris, (TAM_CARA, TAM_CARA), interpolation=cv2.INTER_AREA)
    return extraer(_clahe.apply(cara))
