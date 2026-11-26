"""Pruebas de preprocesado + LBP + comparación. Ejecutar con: python -m pytest -q"""
import cv2
import numpy as np
import pytest
from skimage import data

from detector import detectar
from lbp import N_CASILLAS, TAM_TEMPLATE, ErrorTemplate, comparar, extraer, validar_template
from pipeline import obtener_template, obtener_template_sd
from preprocesado import (OJO_DER_DESTINO, OJO_IZQ_DESTINO, ErrorPreprocesado, preprocesar,
                          preprocesar_sin_detector)

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
    """Tras alinear, los ojos deben quedar cerca de su posición destino."""
    cara = preprocesar(CARA, *detectar(CARA))
    # Para volver a detectar, se reconstruye una cara "normal" sin Tan-Triggs
    caja, oi, od = detectar(CARA)
    from preprocesado import _matriz_semejanza
    gris = cv2.warpAffine(CARA, _matriz_semejanza(oi, od, 128), (128, 128))
    grande = cv2.resize(gris, (512, 512))
    _, oi2, od2 = detectar(np.pad(grande, ((128, 128), (128, 128), (0, 0))))
    oi2 = (np.array(oi2) - 128) / 4
    od2 = (np.array(od2) - 128) / 4
    assert np.allclose(oi2, np.array(OJO_IZQ_DESTINO) * 128, atol=6)
    assert np.allclose(od2, np.array(OJO_DER_DESTINO) * 128, atol=6)
    assert cara.std() > 20


def test_ojos_absurdos_usa_la_caja():
    caja, _, _ = detectar(CARA)
    cara = preprocesar(CARA, caja, (0, 0), (1, 1))      # ojos falsos -> no revienta
    assert cara.shape == (128, 128)


def test_imagen_lisa_se_rechaza():
    with pytest.raises(ErrorPreprocesado):
        preprocesar_sin_detector(np.full((200, 200, 3), 128, np.uint8))


# ---------------------------------------------------------------- extractor
def test_template_formato():
    t = obtener_template(CARA)
    assert t.shape == (TAM_TEMPLATE,) == (5900,) and t.dtype == np.float32
    validar_template(t)


def test_mismo_template_misma_foto():
    assert comparar(obtener_template(CARA), obtener_template(CARA)) == 0.0


def test_genuino_mas_cerca_que_otra_zona():
    t = obtener_template(CARA)
    t_var = obtener_template(variante(CARA, angulo=5, brillo=0.8, desplaz=4))
    # misma foto pero "alineando" con los ojos sobre la boca -> otra textura distinta
    t_otra = extraer(preprocesar(CARA, (178, 63, 90, 113), (200, 160), (247, 160)))
    assert comparar(t, t_var) < comparar(t, t_otra)


def test_invariante_a_brillo_global():
    t1 = obtener_template(CARA)
    t2 = obtener_template(variante(CARA, brillo=0.6))
    assert comparar(t1, t2) < 0.15


def test_template_simetrico_igual_en_espejo():
    """El template es simétrico: la foto en espejo da (casi) el mismo template."""
    t1 = obtener_template(CARA)
    t2 = obtener_template(cv2.flip(CARA, 1))
    assert comparar(t1, t2) < 0.15


def test_extraer_rechaza_entradas_malas():
    for mala in (np.zeros((64, 64), np.uint8), np.zeros((128, 128, 3), np.uint8),
                 np.zeros((128, 128), np.float32), "cara"):
        with pytest.raises(ErrorTemplate):
            extraer(mala)


def test_sin_detector_coherente_con_detector():
    """Con la caja del detector, la vía _sd da un template parecido al de la vía normal."""
    caja, _, _ = detectar(CARA)
    x, y, w, h = caja
    t_sd = obtener_template_sd(CARA[y:y + h, x:x + w])
    assert comparar(obtener_template(CARA), t_sd) < 0.3


# ---------------------------------------------------------------- templates manipulados (_t)
@pytest.mark.parametrize("malo", [
    np.zeros(TAM_TEMPLATE),                               # histogramas vacíos
    np.ones(TAM_TEMPLATE),                                # no normalizado
    np.zeros(10),                                         # tamaño incorrecto
    np.full(TAM_TEMPLATE, np.nan),                        # NaN
    np.full(TAM_TEMPLATE, np.inf),                        # infinito
    -np.full(TAM_TEMPLATE, 1 / N_CASILLAS),               # negativos
    "texto", None, [[1, 2], [3, 4]], b"\x00" * 100,
])
def test_templates_manipulados_se_rechazan(malo):
    with pytest.raises(ErrorTemplate):
        comparar(obtener_template(CARA), malo)


def test_rango_de_distancia():
    t = obtener_template(CARA)
    uniforme = np.full(TAM_TEMPLATE, 1 / N_CASILLAS, dtype=np.float32)
    assert 0 <= comparar(t, uniforme) <= 2
