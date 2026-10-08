"""Pruebas del detector. Ejecutar con: python -m pytest -q   (o: python test_detector.py)"""
import cv2
import numpy as np
import pytest
from skimage import data

from detector import (DetectorCaras, ImagenInvalidaError, SinCaraError,
                      VariasCarasError, cargar_imagen, detectar)

CARA = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)   # 512x512, una cara
det = DetectorCaras()


def test_interfaz_grupo():
    caja, oi, od = detectar(CARA)
    x, y, w, h = caja
    assert w > 50 and h > 50
    assert oi[0] < od[0]                       # ojo_izq a la izquierda en la imagen
    for px, py in (oi, od):
        assert x <= px <= x + w and y <= py <= y + h


def test_ruta_y_bytes(tmp_path):
    p = tmp_path / "cara.jpg"
    cv2.imwrite(str(p), CARA)
    assert detectar(str(p))[0] == detectar(p.read_bytes())[0]


def test_sin_cara():
    ruido = np.random.default_rng(0).integers(0, 256, (400, 400, 3), dtype=np.uint8)
    with pytest.raises(SinCaraError):
        detectar(ruido)
    with pytest.raises(SinCaraError):
        detectar(np.full((300, 300, 3), 128, np.uint8))


def test_varias_caras_elige_la_mayor():
    grande = CARA
    pequena = cv2.resize(CARA, (256, 256))
    lienzo = np.zeros((512, 512 + 256, 3), np.uint8)
    lienzo[:, :512] = grande
    lienzo[128:384, 512:] = pequena
    caras = det.detectar_todas(lienzo)
    assert len(caras) == 2
    assert det.detectar_cara(lienzo).caja[0] < 512          # la grande está a la izquierda
    with pytest.raises(VariasCarasError):
        det.detectar_cara(lienzo, varias="error")


def test_formatos_de_entrada():
    gris = cv2.cvtColor(CARA, cv2.COLOR_BGR2GRAY)
    bgra = cv2.cvtColor(CARA, cv2.COLOR_BGR2BGRA)
    flotante = CARA.astype(np.float32) / 255.0
    for img in (gris, bgra, flotante):
        assert detectar(img)


def test_imagen_grande_coordenadas_correctas():
    grande = cv2.resize(CARA, (3000, 3000))
    caja_g = np.array(detectar(grande)[0])
    caja_n = np.array(detectar(CARA)[0]) * (3000 / 512)
    assert np.allclose(caja_g, caja_n, rtol=0.1, atol=20)


@pytest.mark.parametrize("angulo", [-45, -30, -15, 15, 30, 45])
def test_cara_inclinada_se_detecta(angulo):
    M = cv2.getRotationMatrix2D((215, 120), angulo, 1.0)    # girar sobre la cara
    girada = cv2.warpAffine(CARA, M, (512, 512))
    assert det.detectar_cara(girada).confianza > 0.8


@pytest.mark.xfail(reason="LIMITACIÓN CONOCIDA: los ojos de YuNet infravaloran la inclinación "
                          "de la cabeza a partir de ~10°", strict=True)
def test_angulo_ojos_con_cara_inclinada():
    M = cv2.getRotationMatrix2D((215, 120), 30, 1.0)
    girada = cv2.warpAffine(CARA, M, (512, 512))
    esperado = det.detectar_cara(CARA).angulo_ojos - 30
    assert abs(det.detectar_cara(girada).angulo_ojos - esperado) < 8


def test_cara_en_el_borde():
    recortada = CARA[40:, 140:]                   # la cara queda pegada al borde
    x, y, w, h = detectar(recortada)[0]
    H, W = recortada.shape[:2]
    assert x >= 0 and y >= 0 and x + w <= W and y + h <= H


@pytest.mark.parametrize("entrada", [
    b"", b"esto no es una imagen", 12345, None,
    np.zeros((10, 10, 3), np.uint8),               # demasiado pequeña
    np.zeros((0, 0, 3), np.uint8),
    np.full((100, 100, 3), np.nan, np.float32),
    "/no/existe.jpg",
])
def test_entradas_invalidas(entrada):
    with pytest.raises(ImagenInvalidaError):
        cargar_imagen(entrada)


def test_jpeg_corrupto():
    ok, buf = cv2.imencode(".jpg", CARA)
    truncado = buf.tobytes()[:200]
    with pytest.raises((ImagenInvalidaError, SinCaraError)):
        detectar(truncado)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
