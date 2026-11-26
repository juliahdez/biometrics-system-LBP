"""
Sistema de reconocimiento facial con LBP: los 9 métodos de la práctica.

Uso rápido (funciones de módulo sobre la base de datos por defecto):

    from sistema import inscribir, verificar, identificar
    inscribir("Angelina Jolie", "fotos/angelina.jpg")
    verificar("Angelina Jolie", "otra_foto.jpg")     # -> True / False
    identificar("otra_foto.jpg")                      # -> "Angelina Jolie" / None

O con un objeto (para usar otra base de datos u otro umbral):

    from sistema import SistemaBiometrico
    s = SistemaBiometrico(ruta_bd="otra/bd.csv", umbral=0.33)

| Método                     | Detector | Extractor | Entrada              |
|----------------------------|:--------:|:---------:|----------------------|
| inscribir(id, foto)        |    sí    |    sí     | foto completa        |
| verificar(id, foto)        |    sí    |    sí     | foto completa        |
| identificar(foto)          |    sí    |    sí     | foto completa        |
| inscribir_sd(id, foto)     |    no    |    sí     | foto ya recortada    |
| verificar_sd(id, foto)     |    no    |    sí     | foto ya recortada    |
| identificar_sd(foto)       |    no    |    sí     | foto ya recortada    |
| inscribir_t(id, template)  |    no    |    no     | template             |
| verificar_t(id, template)  |    no    |    no     | template             |
| identificar_t(template)    |    no    |    no     | template             |

`foto`: ruta, bytes del fichero o array numpy BGR.
`template`: array/lista de 5900 valores, o ruta a un .npy / .csv / .txt.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from basedatos import BaseDatos, BaseDatosComprometidaError, ClaveError, clave_id  # noqa: F401
from bloqueo import BloqueoError, bloqueo  # noqa: F401
from detector import ErrorDetector, ImagenInvalidaError, SinCaraError  # noqa: F401
from lbp import ErrorTemplate, TAM_TEMPLATE, distancias, validar_template
from pipeline import obtener_template, obtener_template_sd

log = logging.getLogger("biometria")

MAX_BYTES_TEMPLATE = 1024 * 1024
EXTENSIONES_TEMPLATE = {".npy", ".csv", ".txt"}
# Solo alfabeto latino (con tildes, ñ, ç, ü...). Antes se aceptaba cualquier letra
# Unicode y se podía inscribir "Аngelina Jolie" con una A cirílica, idéntica a la vista.
_LETRA = "A-Za-z0-9\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF\u0100-\u017F"
_PATRON_ID = re.compile(rf"[{_LETRA}][{_LETRA} .'_\-]*")


# --------------------------------------------------------------------------- #
# Errores
# --------------------------------------------------------------------------- #
class ErrorBiometrico(Exception):
    """Error base del sistema."""


class IdInvalidoError(ErrorBiometrico):
    pass


class YaInscritoError(ErrorBiometrico):
    pass


class InscripcionRechazadaError(ErrorBiometrico):
    """El template parece un duplicado de otra persona o un template 'lobo'."""


@dataclass(frozen=True)
class Resultado:
    aceptado: bool
    id: str | None           # id verificado / identificado (None si no hay)
    distancia: float | None  # distancia al mejor template (None si no hay con quién comparar)
    motivo: str = ""         # por qué se ha rechazado (para depurar y para el informe)


# --------------------------------------------------------------------------- #
# Validación de entradas
# --------------------------------------------------------------------------- #
def validar_id(id_) -> str:
    if not isinstance(id_, str):
        raise IdInvalidoError("El id debe ser un texto")
    id_ = unicodedata.normalize("NFC", id_)
    if re.search(r"[^\S ]", id_):        # tabuladores, saltos de línea, espacios raros
        raise IdInvalidoError("El id solo puede llevar espacios normales")
    id_ = re.sub(r" +", " ", id_).strip()
    if not id_ or len(id_) > config.LONGITUD_MAX_ID:
        raise IdInvalidoError(f"El id debe tener entre 1 y {config.LONGITUD_MAX_ID} caracteres")
    # Empieza por letra o número (evita inyección de fórmulas en el .csv: =, +, -, @)
    # y solo contiene letras latinas, números, espacios y . _ ' -
    if not _PATRON_ID.fullmatch(id_):
        raise IdInvalidoError("El id solo puede tener letras (alfabeto latino), números, espacios "
                              "y . _ ' - y debe empezar por letra o número")
    if not clave_id(id_):
        raise IdInvalidoError("El id debe contener alguna letra o número")
    return id_


def cargar_template(template) -> np.ndarray:
    """Acepta un vector o una ruta a .npy / .csv / .txt y lo valida."""
    if isinstance(template, (str, Path)):
        ruta = Path(template)
        if ruta.suffix.lower() not in EXTENSIONES_TEMPLATE:
            raise ErrorTemplate("El template debe ser un fichero .npy, .csv o .txt")
        if not ruta.is_file():
            raise ErrorTemplate(f"No existe el fichero de template {ruta}")
        if ruta.stat().st_size > MAX_BYTES_TEMPLATE:
            raise ErrorTemplate("Fichero de template demasiado grande")
        try:
            if ruta.suffix.lower() == ".npy":
                template = np.load(ruta, allow_pickle=False)   # nunca pickle: ejecutaría código
            else:
                texto = ruta.read_text(encoding="utf-8").replace(",", " ").replace(";", " ")
                template = np.array(texto.split(), dtype=np.float64)
        except Exception:  # noqa: BLE001 - ValueError, OSError, OverflowError (cabecera .npy falsa)...
            # Sin el detalle de la excepción: incluía el contenido del fichero leído
            # (con verificar_t(id, "/etc/passwd") se veía la primera línea).
            raise ErrorTemplate("No se pudo leer el template: el fichero no tiene el formato esperado") from None
    return validar_template(template).astype(np.float32)


# --------------------------------------------------------------------------- #
# Sistema
# --------------------------------------------------------------------------- #
class SistemaBiometrico:
    def __init__(self, ruta_bd=None, ruta_clave=None, umbral: float | None = None,
                 ruta_versiones=None, limitar_intentos: bool = True):
        """limitar_intentos=False solo para los scripts de evaluación (miden miles de intentos)."""
        ruta_clave = Path(ruta_clave or config.RUTA_CLAVE)
        por_defecto = ruta_clave == config.RUTA_CLAVE
        if ruta_versiones is None:
            # con la clave por defecto se usan las rutas de config; con otra, junto a esa clave
            ruta_versiones = config.RUTA_VERSIONES if por_defecto else ruta_clave.with_name("versiones.json")
        self.bd = BaseDatos(Path(ruta_bd or config.RUTA_BD), ruta_clave, Path(ruta_versiones))
        self.ruta_intentos = config.RUTA_INTENTOS if por_defecto else ruta_clave.with_name("intentos.json")
        self.limitar_intentos = limitar_intentos
        self.umbral = config.UMBRAL if umbral is None else umbral
        self.ultimo: Resultado | None = None   # detalle de la última verificación/identificación
        ref = Path(config.RUTA_CENTRO_REFERENCIA)
        self._centro_ref = validar_template(np.load(ref, allow_pickle=False)) if ref.is_file() else None

    # ------------------------------------------------------------ umbral
    @property
    def umbral(self) -> float:
        return self._umbral

    @umbral.setter
    def umbral(self, valor) -> None:
        valor = float(valor)
        if not 0 < valor <= config.UMBRAL:
            # Solo se permite ser MÁS estricto que el umbral calibrado
            raise ValueError(f"El umbral debe estar entre 0 y {config.UMBRAL}")
        self._umbral = valor

    # ------------------------------------------------------------ límite de intentos
    def _leer_intentos(self) -> dict:
        try:
            datos = json.loads(Path(self.ruta_intentos).read_text(encoding="utf-8"))
            return datos if isinstance(datos, dict) else {}
        except FileNotFoundError:
            return {}
        except (ValueError, OSError):
            log.warning("ALERTA %s ilegible: se reinicia el contador de intentos", self.ruta_intentos)
            return {}

    def _guardar_intentos(self, datos: dict) -> None:
        ruta = Path(self.ruta_intentos)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        tmp = ruta.with_name(f".{ruta.name}.tmp")
        tmp.write_text(json.dumps(datos), encoding="utf-8")
        tmp.replace(ruta)

    def _bloqueado_hasta(self, clave: str) -> float:
        if not self.limitar_intentos:
            return 0.0
        with bloqueo(Path(self.ruta_intentos).with_suffix(".lock")):
            return float(self._leer_intentos().get(clave, {}).get("hasta", 0))

    def _registrar(self, clave: str, acierto: bool) -> None:
        if not self.limitar_intentos:
            return
        with bloqueo(Path(self.ruta_intentos).with_suffix(".lock")):
            datos = self._leer_intentos()
            e = datos.get(clave, {"fallos": 0, "bloqueos": 0, "hasta": 0})
            if acierto:
                e = {"fallos": 0, "bloqueos": 0, "hasta": 0}
            else:
                e["fallos"] = e.get("fallos", 0) + 1
                if e["fallos"] >= config.MAX_FALLOS_SEGUIDOS:
                    e["bloqueos"] = e.get("bloqueos", 0) + 1
                    duracion = config.BLOQUEO_SEGUNDOS * 2 ** min(e["bloqueos"] - 1, 10)
                    e["hasta"] = time.time() + duracion
                    e["fallos"] = 0
                    log.warning("ALERTA %d fallos seguidos: id bloqueado %d s", config.MAX_FALLOS_SEGUIDOS, duracion)
            datos[clave] = e
            self._guardar_intentos(datos)

    # ------------------------------------------------------------ defensa anti-lobo
    def _motivo_lobo(self, template, matriz: np.ndarray | None, inscripcion: bool = False) -> str:
        """'' si el template es normal; si parece un template 'lobo', el motivo."""
        centros = []
        if self._centro_ref is not None:
            centros.append(self._centro_ref)
        if matriz is not None and len(matriz) >= config.MIN_INSCRITOS_PARA_CENTRO:
            centros.append(matriz.mean(axis=0))
        if centros:
            d = float(distancias(template, np.stack(centros)).min())
            if d <= config.UMBRAL_LOBO:
                return f"template demasiado cerca de la media de todas las caras ({d:.3f}): posible template lobo"
        if inscripcion and matriz is not None and len(matriz) >= config.MIN_INSCRITOS_PARA_COINCIDENCIAS:
            frac = float((distancias(template, matriz) <= self.umbral).mean())
            if frac > config.MAX_FRACCION_COINCIDENCIAS:
                return f"el template coincide con el {100 * frac:.0f} % de los inscritos: posible template lobo"
        return ""

    # ------------------------------------------------------------ núcleo
    def _inscribir(self, id_, template) -> bool:
        id_ = validar_id(id_)
        template = validar_template(template).astype(np.float32)
        with self.bd.transaccion():      # comprobar y escribir sin que otro proceso se cuele
            return self._inscribir_bloqueado(id_, template)

    def _inscribir_bloqueado(self, id_: str, template: np.ndarray) -> bool:
        filas = self.bd.cargar()
        parecido = next((i for i, _ in filas if clave_id(i) == clave_id(id_)), None)
        if parecido is not None:
            # No se permite reinscribir (ni con un nombre que solo cambia en mayúsculas,
            # tildes o separadores): si no, cualquiera podría suplantar una identidad
            raise YaInscritoError(f"'{id_}' ya está inscrito (como '{parecido}')")
        if len(filas) >= config.MAX_INSCRITOS:
            raise InscripcionRechazadaError(f"Se ha alcanzado el máximo de {config.MAX_INSCRITOS} inscritos")
        matriz = np.stack([t for _, t in filas]) if filas else None
        if matriz is not None:
            d = distancias(template, matriz)
            k = int(np.argmin(d))
            if d[k] <= config.UMBRAL_DUPLICADO:
                raise InscripcionRechazadaError(
                    f"La cara es prácticamente idéntica a la de '{filas[k][0]}' (distancia {d[k]:.3f})")
        motivo = self._motivo_lobo(template, matriz, inscripcion=True)
        if motivo:
            log.warning("ALERTA inscripción rechazada (%s): %s", id_, motivo)
            raise InscripcionRechazadaError(motivo)
        self.bd.anadir(id_, template)
        log.info("inscrito %s", id_)
        return True

    def _verificar(self, id_, template) -> bool:
        id_ = validar_id(id_)
        template = validar_template(template)
        filas = self.bd.cargar()
        clave = clave_id(id_)
        plantillas = [t for i, t in filas if clave_id(i) == clave]
        if not plantillas:
            # Mismo resultado que un rechazo normal: no se revela si el id existe
            self.ultimo = Resultado(False, None, None, "id no inscrito")
            return False
        hasta = self._bloqueado_hasta(clave)
        if hasta > time.time():
            # Para quien llama es un rechazo normal: no se le da ninguna pista
            self.ultimo = Resultado(False, None, None, f"id bloqueado {hasta - time.time():.0f} s más por fallos seguidos")
            return False
        motivo = self._motivo_lobo(template, np.stack([t for _, t in filas]))
        if motivo:
            log.warning("ALERTA verificación de %s rechazada: %s", id_, motivo)
            self.ultimo = Resultado(False, None, None, motivo)
            self._registrar(clave, False)
            return False
        d = float(distancias(template, np.stack(plantillas)).min())
        aceptado = d <= self.umbral
        self._registrar(clave, aceptado)
        self.ultimo = Resultado(aceptado, id_ if aceptado else None, d,
                                "" if aceptado else "distancia mayor que el umbral")
        log.info("verificar %s -> %s (%.4f)", id_, aceptado, d)
        return aceptado

    def _identificar(self, template) -> str | None:
        template = validar_template(template)
        filas = self.bd.cargar()
        if not filas:
            self.ultimo = Resultado(False, None, None, "base de datos vacía")
            return None
        matriz = np.stack([t for _, t in filas])
        motivo = self._motivo_lobo(template, matriz)
        if motivo:
            log.warning("ALERTA identificación rechazada: %s", motivo)
            self.ultimo = Resultado(False, None, None, motivo)
            return None
        d = distancias(template, matriz)
        orden = np.argsort(d)
        k = int(orden[0])
        motivo = ""
        if d[k] > self.umbral:
            motivo = "nadie por debajo del umbral"
        elif len(d) > 1 and d[orden[1]] - d[k] < config.MARGEN_IDENTIFICACION:
            motivo = (f"resultado dudoso: '{filas[k][0]}' y '{filas[int(orden[1])][0]}' "
                      f"están casi igual de cerca ({d[k]:.4f} / {d[orden[1]]:.4f})")
        aceptado = not motivo
        id_ = filas[k][0] if aceptado else None
        self.ultimo = Resultado(aceptado, id_, float(d[k]), motivo)
        log.info("identificar -> %s (%.4f)", id_, d[k])
        return id_

    # ------------------------------------------------------------ con detector
    def inscribir(self, id, foto) -> bool:
        return self._inscribir(id, obtener_template(foto))

    def verificar(self, id, foto) -> bool:
        return self._verificar(id, obtener_template(foto))

    def identificar(self, foto) -> str | None:
        return self._identificar(obtener_template(foto))

    # ------------------------------------------------------------ sin detector
    def inscribir_sd(self, id, foto) -> bool:
        return self._inscribir(id, obtener_template_sd(foto))

    def verificar_sd(self, id, foto) -> bool:
        return self._verificar(id, obtener_template_sd(foto))

    def identificar_sd(self, foto) -> str | None:
        return self._identificar(obtener_template_sd(foto))

    # ------------------------------------------------------------ con template
    def inscribir_t(self, id, template) -> bool:
        return self._inscribir(id, cargar_template(template))

    def verificar_t(self, id, template) -> bool:
        return self._verificar(id, cargar_template(template))

    def identificar_t(self, template) -> str | None:
        return self._identificar(cargar_template(template))

    # ------------------------------------------------------------ utilidades
    def listar(self) -> list[str]:
        return [i for i, _ in self.bd.cargar()]


def extraer_template(foto, sin_detector: bool = False) -> np.ndarray:
    """Template de una foto (para probar los métodos _t)."""
    return obtener_template_sd(foto) if sin_detector else obtener_template(foto)


# --------------------------------------------------------------------------- #
# Funciones de módulo sobre el sistema por defecto
# --------------------------------------------------------------------------- #
_sistema: SistemaBiometrico | None = None


def _s() -> SistemaBiometrico:
    global _sistema
    if _sistema is None:
        _sistema = SistemaBiometrico()
    return _sistema


def inscribir(id, foto): return _s().inscribir(id, foto)
def verificar(id, foto): return _s().verificar(id, foto)
def identificar(foto): return _s().identificar(foto)
def inscribir_sd(id, foto): return _s().inscribir_sd(id, foto)
def verificar_sd(id, foto): return _s().verificar_sd(id, foto)
def identificar_sd(foto): return _s().identificar_sd(foto)
def inscribir_t(id, template): return _s().inscribir_t(id, template)
def verificar_t(id, template): return _s().verificar_t(id, template)
def identificar_t(template): return _s().identificar_t(template)


__all__ = [
    "SistemaBiometrico", "Resultado", "extraer_template",
    "inscribir", "verificar", "identificar",
    "inscribir_sd", "verificar_sd", "identificar_sd",
    "inscribir_t", "verificar_t", "identificar_t",
    "ErrorBiometrico", "IdInvalidoError", "YaInscritoError", "InscripcionRechazadaError",
    "ErrorDetector", "SinCaraError", "ImagenInvalidaError", "ErrorTemplate",
    "BaseDatosComprometidaError", "ClaveError", "BloqueoError",
]
