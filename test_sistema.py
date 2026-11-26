"""Pruebas del sistema completo: 9 métodos, base de datos y defensas.

Ejecutar con: python -m pytest -q
"""
import csv
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage import data

import config
from basedatos import BaseDatosComprometidaError, ClaveError
from detector import ImagenInvalidaError, SinCaraError, detectar
from lbp import N_CASILLAS, TAM_TEMPLATE, ErrorTemplate
from sistema import (IdInvalidoError, InscripcionRechazadaError, SistemaBiometrico,
                     YaInscritoError, extraer_template)

ASTRO = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
(_X, _Y, _W, _H), _, _ = detectar(ASTRO)
ASTRO_CARA = ASTRO[_Y:_Y + _H, _X:_X + _W]
CAMARA = cv2.cvtColor(data.coffee(), cv2.COLOR_RGB2BGR)   # taza de café: sin cara


def caras_falsas(n, semilla=0):
    """Templates válidos y distintos entre sí, lejos del centro (como caras reales)."""
    rng = np.random.default_rng(semilla)
    out = []
    for _ in range(n):
        h = rng.dirichlet(np.full(N_CASILLAS, 0.3), size=100)
        out.append(h.ravel().astype(np.float32))
    return out


@pytest.fixture
def s(tmp_path):
    return SistemaBiometrico(ruta_bd=tmp_path / "bd.csv", ruta_clave=tmp_path / "clave.key")


# ---------------------------------------------------------------- 9 métodos
def test_inscribir_verificar_identificar(s):
    assert s.inscribir("Eileen Collins", ASTRO)
    assert s.verificar("Eileen Collins", ASTRO) is True
    assert s.identificar(ASTRO) == "Eileen Collins"
    assert s.ultimo.distancia == 0.0


def test_metodos_sd(s):
    assert s.inscribir_sd("Eileen", ASTRO_CARA)
    assert s.verificar_sd("Eileen", ASTRO_CARA)
    assert s.identificar_sd(ASTRO_CARA) == "Eileen"
    # la vía normal también la reconoce con la foto completa
    assert s.verificar("Eileen", ASTRO)


def test_metodos_t(s, tmp_path):
    t = extraer_template(ASTRO)
    assert s.inscribir_t("Eileen", t)
    assert s.verificar_t("Eileen", t)
    assert s.identificar_t(t) == "Eileen"
    np.save(tmp_path / "t.npy", t)
    np.savetxt(tmp_path / "t.csv", t[None, :], delimiter=",")
    assert s.verificar_t("Eileen", tmp_path / "t.npy")
    assert s.identificar_t(str(tmp_path / "t.csv")) == "Eileen"


def test_impostor_rechazado(s):
    otros = caras_falsas(1)
    s.inscribir_t("Otro", otros[0])
    assert s.verificar("Otro", ASTRO) is False
    assert s.identificar(ASTRO) is None


def test_foto_sin_cara(s):
    with pytest.raises(SinCaraError):
        s.inscribir("X", CAMARA)
    with pytest.raises(SinCaraError):
        s.verificar("X", CAMARA)


def test_id_no_inscrito_es_un_rechazo_normal(s):
    s.inscribir("Eileen", ASTRO)
    assert s.verificar("Nadie", ASTRO) is False


# ---------------------------------------------------------------- reglas de inscripción
def test_no_se_puede_reinscribir(s):
    s.inscribir("Eileen", ASTRO)
    with pytest.raises(YaInscritoError):
        s.inscribir_t("eileen", caras_falsas(1)[0])          # ni cambiando mayúsculas


def test_misma_cara_con_otro_nombre(s):
    s.inscribir("Eileen", ASTRO)
    with pytest.raises(InscripcionRechazadaError):
        s.inscribir("Suplantador", ASTRO)


@pytest.mark.parametrize("id_malo", [
    "", " ", "=HYPERLINK(\"http://x\")", "+1", "-1", "@SUM(A1)", "a,b", "a\nb",
    "a\"b", "x" * 65, 123, None, "_oculto", "../../etc", "a\tb", "a\u00a0b",
    "\u0410ngelina Jolie",                       # A cirílica: se ve igual que la latina
    "Ελένη", "李", "...", "a\u200bb",
])
def test_ids_invalidos(s, id_malo):
    with pytest.raises(IdInvalidoError):
        s.inscribir_t(id_malo, caras_falsas(1)[0])


def test_ids_validos(s):
    for i, id_ in enumerate(["Penélope Cruz", "Shaquille O'Neal", "Jean-Luc 2", "Ñandú_9"]):
        s.inscribir_t(id_, caras_falsas(1, semilla=i)[0])
    assert len(s.listar()) == 4


# ---------------------------------------------------------------- template lobo
def test_template_lobo_rechazado(s):
    caras = caras_falsas(12)
    for i, t in enumerate(caras):
        s.inscribir_t(f"P{i}", t)
    lobo = np.mean(caras, axis=0)
    assert s.verificar_t("P0", lobo) is False and "lobo" in s.ultimo.motivo
    assert s.identificar_t(lobo) is None
    with pytest.raises(InscripcionRechazadaError):
        s.inscribir_t("lobo", lobo)


# ---------------------------------------------------------------- base de datos
def test_bd_no_contiene_templates_en_claro(s, tmp_path):
    t = extraer_template(ASTRO)
    s.inscribir_t("Eileen", t)
    contenido = (tmp_path / "bd.csv").read_bytes()
    assert t.tobytes()[:64] not in contenido
    filas = list(csv.reader(io.StringIO(contenido.decode())))
    assert filas[0] == ["id", "template"] and filas[1][0] == "Eileen"


def _reescribir(ruta, nuevo_texto):
    ruta.write_text(nuevo_texto, encoding="utf-8")


def test_bd_modificada_se_detecta(s, tmp_path):
    s.inscribir("Eileen", ASTRO)
    s.inscribir_t("Otro", caras_falsas(1)[0])
    ruta = tmp_path / "bd.csv"
    original = ruta.read_text()
    cab, l1, l2, firma = original.splitlines()
    assert firma.startswith("#firma,")
    (n1, t1), (n2, t2) = l1.split(",", 1), l2.split(",", 1)
    otro_char = "A" if t1[-5] != "A" else "B"
    casos = {
        "cambiar un carácter": [cab, f"{n1},{t1[:-5]}{otro_char}{t1[-4:]}", l2, firma],
        "intercambiar templates": [cab, f"{n1},{t2}", f"{n2},{t1}", firma],
        "borrar una fila": [cab, l1, firma],
        "duplicar una fila": [cab, l1, l2, l2, firma],
        "reordenar": [cab, l2, l1, firma],
        "quitar la firma": [cab, l1, l2],
        "firma inventada": [cab, l1, l2, "#firma,99:" + "0" * 64],
    }
    for nombre, lineas in casos.items():
        _reescribir(ruta, "\n".join(lineas) + "\n")
        with pytest.raises(BaseDatosComprometidaError):
            s.listar()
    _reescribir(ruta, original)
    assert len(s.listar()) == 2


def test_bd_comprometida_no_permite_escribir(s, tmp_path):
    s.inscribir("Eileen", ASTRO)
    ruta = tmp_path / "bd.csv"
    ruta.write_text(ruta.read_text().replace("#firma,1:", "#firma,1:0"))
    with pytest.raises(BaseDatosComprometidaError):
        s.inscribir_t("Otro", caras_falsas(1)[0])


def test_crlf_de_git_en_windows_no_rompe_la_bd(s, tmp_path):
    s.inscribir_t("A", caras_falsas(1)[0])
    ruta = tmp_path / "bd.csv"
    ruta.write_bytes(ruta.read_bytes().replace(b"\n", b"\r\n"))
    assert s.listar() == ["A"]
    clave = (tmp_path / "clave.key").read_bytes()
    assert len(clave.strip()) == 64 and all(chr(c) in "0123456789abcdef" for c in clave.strip())
    (tmp_path / "clave.key").write_bytes(clave.strip() + b"\r\n")       # clave con CRLF
    assert SistemaBiometrico(ruta_bd=ruta, ruta_clave=tmp_path / "clave.key").listar() == ["A"]


def test_corte_a_mitad_de_escritura_no_rompe_la_bd(s, tmp_path, monkeypatch):
    import basedatos
    s.inscribir_t("A", caras_falsas(1, semilla=1)[0])
    original = basedatos._escritura_atomica
    for corte_en in (1, 2):          # 1: antes de escribir el .csv  2: entre .csv y contador
        llamadas = [0]

        def con_corte(ruta, contenido, corte_en=corte_en):
            llamadas[0] += 1
            if llamadas[0] == corte_en:
                raise KeyboardInterrupt
            original(ruta, contenido)
        monkeypatch.setattr(basedatos, "_escritura_atomica", con_corte)
        with pytest.raises(KeyboardInterrupt):
            s.inscribir_t(f"B{corte_en}", caras_falsas(1, semilla=10 + corte_en)[0])
        monkeypatch.setattr(basedatos, "_escritura_atomica", original)
        nuevo = SistemaBiometrico(ruta_bd=tmp_path / "bd.csv", ruta_clave=tmp_path / "clave.key")
        assert "A" in nuevo.listar()          # la BD sigue funcionando
    assert len(list(tmp_path.glob(".*.tmp"))) == 0


def test_varias_bd_con_la_misma_clave(s, tmp_path):
    s.inscribir_t("A", caras_falsas(1)[0])
    otra = SistemaBiometrico(ruta_bd=tmp_path / "otra" / "bd2.csv", ruta_clave=tmp_path / "clave.key")
    assert otra.listar() == []
    otra.inscribir_t("B", caras_falsas(1, semilla=3)[0])
    assert s.listar() == ["A"] and otra.listar() == ["B"]


def test_otra_clave_no_sirve(s, tmp_path):
    s.inscribir("Eileen", ASTRO)
    (tmp_path / "otra.key").write_bytes(b"\x01" * 32)
    intruso = SistemaBiometrico(ruta_bd=tmp_path / "bd.csv", ruta_clave=tmp_path / "otra.key")
    with pytest.raises(BaseDatosComprometidaError):
        intruso.listar()


def test_falta_la_clave_con_bd_existente(s, tmp_path):
    s.inscribir("Eileen", ASTRO)
    (tmp_path / "clave.key").unlink()
    with pytest.raises(ClaveError):
        SistemaBiometrico(ruta_bd=tmp_path / "bd.csv", ruta_clave=tmp_path / "clave.key")


# ---------------------------------------------------------------- templates de entrada
def test_template_fichero_pickle_rechazado(s, tmp_path):
    ruta = tmp_path / "malo.npy"
    np.save(ruta, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ErrorTemplate):
        s.verificar_t("x", ruta)


def test_template_fichero_inexistente_o_basura(s, tmp_path):
    with pytest.raises(ErrorTemplate):
        s.identificar_t(tmp_path / "no_existe.npy")
    (tmp_path / "basura.txt").write_text("hola mundo")
    with pytest.raises(ErrorTemplate):
        s.identificar_t(tmp_path / "basura.txt")


def test_umbral_por_defecto():
    assert 0.2 < config.UMBRAL < 0.5
    assert TAM_TEMPLATE == 5900


def test_identificacion_dudosa_no_da_nombre(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "UMBRAL", 2.0)   # que el umbral no sea lo que rechaza
    s = SistemaBiometrico(ruta_bd=tmp_path / "bd.csv", ruta_clave=tmp_path / "clave.key")
    a, b = caras_falsas(2, semilla=7)
    s.inscribir_t("A", a)
    s.inscribir_t("B", b)
    mezcla = (a + b) / 2                         # a la misma distancia de los dos
    assert s.identificar_t(mezcla) is None
    assert "dudoso" in s.ultimo.motivo
    assert s.identificar_t(a) == "A"


def test_rollback_detectado(s, tmp_path):
    s.inscribir_t("A", caras_falsas(1, semilla=1)[0])
    vieja = (tmp_path / "bd.csv").read_bytes()
    s.inscribir_t("B", caras_falsas(1, semilla=2)[0])
    (tmp_path / "bd.csv").write_bytes(vieja)
    with pytest.raises(BaseDatosComprometidaError, match="rollback"):
        s.listar()


def test_borrar_bd_entera_detectado(s, tmp_path):
    s.inscribir_t("A", caras_falsas(1)[0])
    (tmp_path / "bd.csv").unlink()
    with pytest.raises(BaseDatosComprometidaError):
        s.listar()


# ---------------------------------------------------------------- fallos encontrados atacando como el rival
@pytest.mark.parametrize("parecido", ["angelina_jolie", "ANGELINA-JOLIE", "Angelina  Jolie",
                                      "Ángelina Jolie", "Angelına Jolie", "angelina.jolie"])
def test_nombres_casi_iguales_son_el_mismo(s, parecido):
    s.inscribir_t("Angelina Jolie", caras_falsas(1, semilla=3)[0])
    with pytest.raises(YaInscritoError):
        s.inscribir_t(parecido, caras_falsas(1, semilla=4)[0])
    # y se puede verificar con cualquiera de las variantes
    assert s.verificar_t(parecido, caras_falsas(1, semilla=3)[0])


def test_umbral_no_se_puede_subir(s):
    with pytest.raises(ValueError):
        s.umbral = config.UMBRAL + 0.01
    with pytest.raises(ValueError):
        SistemaBiometrico(ruta_bd=s.bd.ruta, umbral=5)
    s.umbral = config.UMBRAL - 0.05                     # más estricto sí
    assert s.umbral < config.UMBRAL


def test_cli_no_permite_cambiar_umbral_ni_bd():
    import biometria
    for opcion in (["--umbral", "5"], ["--bd", "x.csv"], ["--clave", "x.key"], ["--detalle"]):
        with pytest.raises(SystemExit):
            biometria.main(opcion + ["listar"])


def test_bloqueo_tras_fallos_seguidos(s):
    t, otro = caras_falsas(2, semilla=5)
    s.inscribir_t("Victima", t)
    for _ in range(config.MAX_FALLOS_SEGUIDOS):
        assert s.verificar_t("Victima", otro) is False
    # bloqueado: ni siquiera el template correcto entra
    assert s.verificar_t("Victima", t) is False
    assert "bloqueado" in s.ultimo.motivo


def test_acierto_reinicia_los_fallos(s):
    t, otro = caras_falsas(2, semilla=6)
    s.inscribir_t("Victima", t)
    for _ in range(5):
        for _ in range(config.MAX_FALLOS_SEGUIDOS - 1):
            s.verificar_t("Victima", otro)
        assert s.verificar_t("Victima", t) is True


def test_maximo_de_inscritos(s, monkeypatch):
    monkeypatch.setattr(config, "MAX_INSCRITOS", 3)
    for i, t in enumerate(caras_falsas(3, semilla=9)):
        s.inscribir_t(f"P{i}", t)
    with pytest.raises(InscripcionRechazadaError):
        s.inscribir_t("P3", caras_falsas(1, semilla=10)[0])


def test_error_de_template_no_muestra_el_fichero(s, tmp_path):
    secreto = tmp_path / "secreto.txt"
    secreto.write_text("contraseña_super_secreta")
    with pytest.raises(ErrorTemplate) as e:
        s.verificar_t("x", secreto)
    assert "contraseña" not in str(e.value)
    with pytest.raises(ErrorTemplate):
        s.verificar_t("x", tmp_path / "algo.key")          # extensión no permitida


def _inscribir_en_paralelo(args):
    ruta, i = args
    s = SistemaBiometrico(ruta_bd=ruta / "bd.csv", ruta_clave=ruta / "clave.key")
    for k, t in enumerate(caras_falsas(4, semilla=100 + i)):
        s.inscribir_t(f"proc{i} n{k}", t)


def test_inscripciones_simultaneas_no_corrompen_la_bd(tmp_path):
    from multiprocessing import get_context
    with get_context("spawn").Pool(4) as pool:
        pool.map(_inscribir_en_paralelo, [(tmp_path, i) for i in range(4)])
    s = SistemaBiometrico(ruta_bd=tmp_path / "bd.csv", ruta_clave=tmp_path / "clave.key")
    assert len([i for i in s.listar() if i.startswith("proc")]) == 16


# ---------------------------------------------------------------- fallos de la revisión externa
def _png_gris(ancho, alto) -> bytes:
    """PNG gris generado por filas: ocupa poco aunque diga ser enorme."""
    import struct
    import zlib

    def trozo(tipo, datos):
        return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", zlib.crc32(tipo + datos))
    c = zlib.compressobj(9)
    fila = b"\x00" + b"\x80" * ancho
    idat = b"".join(c.compress(fila) for _ in range(alto)) + c.flush()
    return (b"\x89PNG\r\n\x1a\n" + trozo(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, 8, 0, 0, 0, 0))
            + trozo(b"IDAT", idat) + trozo(b"IEND", b""))


@pytest.mark.skipif(not Path("/proc/self/status").exists(), reason="mide la memoria con /proc (Linux)")
def test_bomba_de_descompresion_no_gasta_memoria(tmp_path):
    """PNG de ~600 KB que dice medir 25000x25000: antes gastaba 3,7 GB."""
    import subprocess
    import sys
    (tmp_path / "bomba.png").write_bytes(_png_gris(25000, 25000))
    codigo = (
        "import cv2, sistema\n"          # cv2 primero: el peor caso
        f"try: sistema.extraer_template({str(tmp_path / 'bomba.png')!r})\n"
        "except sistema.ImagenInvalidaError: print('RECHAZADA')\n"
        "print(int([l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]) // 1024)\n")
    out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True,
                         cwd=str(Path(__file__).parent)).stdout.split()
    assert out[0] == "RECHAZADA"
    assert int(out[1]) < 600, f"pico de memoria {out[1]} MB"


def test_entradas_raras_dan_errores_controlados(s, tmp_path):
    s.inscribir_t("A", caras_falsas(1)[0])
    npy = tmp_path / "cabecera_falsa.npy"
    np.save(npy, np.zeros(TAM_TEMPLATE, np.float32))
    npy.write_bytes(npy.read_bytes().replace(b"(5900,), }", b"(999999999999999999999,), }"))
    with pytest.raises(ErrorTemplate):
        s.verificar_t("A", npy)
    for foto in (np.empty((100, 100, 3), dtype=object), np.zeros((100, 100, 3), dtype=bool),
                 np.zeros((100, 100, 3), dtype=np.complex64)):
        with pytest.raises(ImagenInvalidaError):
            s.verificar("A", foto)
