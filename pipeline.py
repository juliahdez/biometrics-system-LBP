"""
Cadena completa: foto -> detector -> preprocesado -> LBP -> template.

    obtener_template(foto)       con detector    (inscribir / verificar / identificar)
    obtener_template_sd(foto)    sin detector    (inscribir_sd / verificar_sd / identificar_sd)
"""
from __future__ import annotations

from detector import cargar_imagen, detectar
from lbp import comparar, extraer  # noqa: F401  (comparar se reexporta)
from preprocesado import preprocesar, preprocesar_sin_detector


def obtener_template(foto):
    img = cargar_imagen(foto)
    caja, ojo_izq, ojo_der = detectar(img)
    return extraer(preprocesar(img, caja, ojo_izq, ojo_der))


def obtener_template_sd(foto):
    """Sin detector: la foto se trata como el recorte de la cara (la caja del detector)."""
    return extraer(preprocesar_sin_detector(foto))
