"""Pruebas de preprocesado + LBP + comparación. Ejecutar con: python -m pytest -q"""
import cv2
import numpy as np
import pytest
from skimage import data

from detector import detectar
from lbp import N_CASILLAS, TAM_TEMPLATE, ErrorTemplate, comparar, extraer, validar_template
from pipeline import obtener_template, obtener_template_sd
from preprocesado import preprocesar

CARA = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


def variante(img, angulo=0.0, brillo=1.0, desplaz=0):
    M = cv2.getRotationMatrix2D((215 + desplaz, 120), angulo, 1.0)
    out = cv2.warpAffine(img, M, img.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
    return np.clip(out.astype(np.float32) * brillo, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- preprocesado
def test_preprocesado_forma_y_tipo():
    cara = preprocesar(CARA, *detectar(CARA))
    assert cara.shape == (128, 128) and cara.dtype == np.uint8


def test_preprocesado_pone_los_ojos_en_su_sitio():
    """Tras alinear, el detector debe encontrar los ojos cerca de (38, 46) y (90, 46)."""
    cara = preprocesar(CARA, *detectar(CARA))
    grande = cv2.resize(cv2.cvtColor(cara, cv2.COLOR_GRAY2BGR), (512, 512))
    _, oi, od = detectar(np.pad(grande, ((128, 128), (128, 128), (0, 0))))
    oi = (np.array(oi) - 128) / 4
    od = (np.array(od) - 128) / 4
    assert np.allclose(oi, (38.4, 46.1), atol=6) and np.allclose(od, (89.6, 46.1), atol=6)


def test_ojos_absurdos_usa_la_caja():
    caja, _, _ = detectar(CARA)
    cara = preprocesar(CARA, caja, (0, 0), (1, 1))      # ojos falsos -> no revienta
    assert cara.shape == (128, 128)


# ---------------------------------------------------------------- extractor
def test_template_formato():
    t = obtener_template(CARA)
    assert t.shape == (TAM_TEMPLATE,) and t.dtype == np.float32
    validar_template(t)


def test_mismo_template_misma_foto():
    assert comparar(obtener_template(CARA), obtener_template(CARA)) == 0.0


def test_genuino_mas_cerca_que_otra_cara():
    t = obtener_template(CARA)
    t_var = obtener_template(variante(CARA, angulo=5, brillo=0.8, desplaz=4))
    # misma foto pero "alineando" con los ojos sobre la boca -> otra cara distinta
    t_otra = extraer(preprocesar(CARA, (178, 63, 90, 113), (200, 160), (247, 160)))
    assert comparar(t, t_var) < comparar(t, t_otra)


def test_invariante_a_brillo_global():
    t1 = obtener_template(CARA)
    t2 = obtener_template(variante(CARA, brillo=0.6))
    assert comparar(t1, t2) < 0.3


def test_extraer_rechaza_entradas_malas():
    for mala in (np.zeros((64, 64), np.uint8), np.zeros((128, 128, 3), np.uint8),
                 np.zeros((128, 128), np.float32), "cara"):
        with pytest.raises(ErrorTemplate):
            extraer(mala)


def test_sin_detector():
    caja, _, _ = detectar(CARA)
    x, y, w, h = caja
    t = obtener_template_sd(CARA[y:y + h, x:x + w])
    assert t.shape == (TAM_TEMPLATE,)


# ---------------------------------------------------------------- templates manipulados (_t)
@pytest.mark.parametrize("malo", [
    np.zeros(TAM_TEMPLATE),                               # histogramas vacíos
    np.ones(TAM_TEMPLATE),                                # no normalizado
    np.zeros(10),                                         # tamaño incorrecto
    np.full(TAM_TEMPLATE, np.nan),                        # NaN
    -obtener_template(CARA),                              # negativos
    "texto", None, [[1, 2], [3, 4]],
])
def test_templates_manipulados_se_rechazan(malo):
    with pytest.raises(ErrorTemplate):
        comparar(obtener_template(CARA), malo)


def test_rango_de_distancia():
    t = obtener_template(CARA)
    uniforme = np.full(TAM_TEMPLATE, 1 / N_CASILLAS, dtype=np.float32)
    assert 0 <= comparar(t, uniforme) <= 2
