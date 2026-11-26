"""
Base de datos de templates en un .csv protegido.

Formato de bd.csv (una fila por persona y la firma como última línea):

    id,template
    Angelina Jolie,<base64( nonce(12 bytes) || AES-256-GCM(template float32) )>
    ...
    #firma,<versión>:<HMAC-SHA256>

Protecciones:
  * Confidencialidad: cada template va cifrado con AES-256-GCM. Quien lea el .csv
    solo ve los nombres, no los templates.
  * Integridad de cada fila: GCM autentica el template junto con su id (datos
    asociados). Cambiar un byte o mover un template a otro nombre se detecta.
  * Integridad del fichero entero: la última línea es un HMAC-SHA256 de todo lo
    anterior. Borrar, duplicar, reordenar o añadir filas se detecta.
  * Un único fichero y un único os.replace por escritura: un corte (Ctrl+C,
    apagón) deja la versión anterior o la nueva, nunca una BD con la firma
    descuadrada. (Antes la firma iba en un fichero aparte y un corte entre las dos
    escrituras dejaba la BD inservible.)
  * Anti-rollback: cada escritura sube un contador de versión que va dentro de la
    firma y también se guarda junto a la clave (secretos/versiones.json, uno por
    BD). Volver a poner una copia antigua (aunque sea válida) se detecta.
  * Tolerante a CRLF: los saltos de línea se normalizan antes de comprobar la firma
    (git en Windows con core.autocrlf convierte \\n en \\r\\n).
  * Si algo no cuadra, el sistema se niega a funcionar (BaseDatosComprometidaError)
    en vez de usar datos que pueden estar manipulados.

La clave maestra vive FUERA del .csv (secretos/clave.key, en hexadecimal, o la
variable de entorno BIOMETRIA_CLAVE). De ella se derivan, con HKDF, una subclave
para cifrar y otra para el HMAC.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import io
import json
import os
import re
import secrets
import string
import unicodedata
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from bloqueo import bloqueo
from lbp import TAM_TEMPLATE


def clave_id(id_: str) -> str:
    """Forma canónica de un id para comparar: sin mayúsculas, tildes ni separadores.

    'Angelina Jolie', 'angelina_jolie', 'ANGELINA-JOLIE' y 'Ángelina  Jolie' son
    el mismo id: así nadie puede inscribir un nombre que a simple vista parezca otro.
    """
    sin_tildes = "".join(c for c in unicodedata.normalize("NFKD", id_)
                         if not unicodedata.combining(c))
    texto = sin_tildes.casefold().translate(_PARECIDAS)
    return re.sub(r"[\s._'\-]+", "", texto)


# Letras latinas que no se descomponen pero se confunden a simple vista con otras
_PARECIDAS = str.maketrans({"ı": "i", "ł": "l", "đ": "d", "ð": "d", "ħ": "h", "ŧ": "t",
                            "ø": "o", "æ": "ae", "œ": "oe", "ŋ": "n", "ĸ": "k", "þ": "th"})

CABECERA = ["id", "template"]
MARCA_FIRMA = "#firma"          # los ids no pueden empezar por '#'
VERSION = b"bd-lbp-v2"


class BaseDatosComprometidaError(Exception):
    """La base de datos ha sido manipulada o no se puede verificar."""


class ClaveError(Exception):
    """No hay clave o la clave no es válida."""


# --------------------------------------------------------------------------- #
# Clave
# --------------------------------------------------------------------------- #
def _leer_clave(crudo: bytes) -> bytes:
    texto = crudo.strip()
    if len(texto) == 64 and all(chr(c) in string.hexdigits for c in texto):
        return bytes.fromhex(texto.decode("ascii"))
    if len(crudo) == 32:                    # formato antiguo: 32 bytes en binario
        return crudo
    raise ClaveError("La clave debe tener 32 bytes (64 caracteres hexadecimales)")


def cargar_o_crear_clave(ruta: Path, crear: bool = True) -> bytes:
    hex_env = os.environ.get("BIOMETRIA_CLAVE")
    if hex_env:
        try:
            return _leer_clave(hex_env.encode("ascii", errors="replace"))
        except ClaveError:
            raise ClaveError("BIOMETRIA_CLAVE debe tener 64 caracteres hexadecimales") from None
    if ruta.is_file():
        return _leer_clave(ruta.read_bytes())
    if not crear:
        raise ClaveError(f"No existe la clave {ruta}")

    ruta.parent.mkdir(parents=True, exist_ok=True)
    # En hexadecimal (texto): git no la estropea con conversiones de fin de línea.
    contenido = (secrets.token_hex(32) + "\n").encode("ascii")
    # Se escribe en un temporal y se enlaza (os.link falla si ya existe): si dos
    # procesos arrancan a la vez, gana uno y el otro usa la clave del ganador.
    tmp = ruta.with_name(f".{ruta.name}.{secrets.token_hex(4)}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(contenido)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, ruta)
    except FileExistsError:
        pass
    except OSError:                          # sistemas de ficheros sin enlaces duros
        try:
            fd = os.open(ruta, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(contenido)
        except FileExistsError:
            pass
    finally:
        tmp.unlink(missing_ok=True)
    return _leer_clave(ruta.read_bytes())


def _subclave(clave: bytes, uso: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=VERSION, info=uso).derive(clave)


# --------------------------------------------------------------------------- #
# Base de datos
# --------------------------------------------------------------------------- #
class BaseDatos:
    def __init__(self, ruta_bd: Path, ruta_clave: Path, ruta_versiones: Path | None = None):
        self.ruta = Path(ruta_bd)
        self.ruta_versiones = (Path(ruta_versiones) if ruta_versiones
                               else Path(ruta_clave).with_name("versiones.json"))
        self.ruta_bloqueo = self.ruta.with_name("." + self.ruta.name + ".lock")
        # Nombre estable de esta BD para el contador (igual en el host y en Docker):
        # "datos/bd.csv". Así varias BD pueden compartir clave sin pisarse el contador.
        self.nombre = f"{self.ruta.parent.name}/{self.ruta.name}"
        # Solo se crea una clave nueva si todavía no hay base de datos: si la hay y
        # falta la clave, es un error (crear otra dejaría la BD ilegible).
        clave = cargar_o_crear_clave(Path(ruta_clave), crear=not self.ruta.exists())
        self._aes = AESGCM(_subclave(clave, b"cifrado-templates"))
        self._clave_mac = _subclave(clave, b"integridad-fichero")
        self._version = 0

    # ---------------------------------------------------------------- utilidades
    @staticmethod
    def _aad(id_: str) -> bytes:
        return VERSION + b"|" + id_.encode("utf-8")

    def _mac(self, cuerpo: bytes, version: int) -> str:
        return hmac.new(self._clave_mac, f"{version}\n".encode() + cuerpo, hashlib.sha256).hexdigest()

    def _leer_versiones(self) -> dict:
        try:
            datos = json.loads(self.ruta_versiones.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (ValueError, OSError):
            raise BaseDatosComprometidaError(f"Contador de versiones ilegible: {self.ruta_versiones}") from None
        if not isinstance(datos, dict):
            raise BaseDatosComprometidaError(f"Contador de versiones ilegible: {self.ruta_versiones}")
        return datos

    def _version_guardada(self) -> int:
        with bloqueo(self.ruta_versiones.with_suffix(".lock")):
            v = self._leer_versiones().get(self.nombre, 0)
        return v if isinstance(v, int) else 0

    def _guardar_version(self, version: int) -> None:
        with bloqueo(self.ruta_versiones.with_suffix(".lock")):
            datos = self._leer_versiones()
            datos[self.nombre] = version
            self.ruta_versiones.parent.mkdir(parents=True, exist_ok=True)
            _escritura_atomica(self.ruta_versiones, json.dumps(datos, indent=1).encode("utf-8"))

    def _cifrar(self, id_: str, template: np.ndarray) -> str:
        nonce = secrets.token_bytes(12)
        datos = np.asarray(template, dtype="<f4").tobytes()
        return base64.b64encode(nonce + self._aes.encrypt(nonce, datos, self._aad(id_))).decode("ascii")

    def _descifrar(self, id_: str, campo: str) -> np.ndarray:
        try:
            crudo = base64.b64decode(campo, validate=True)
            datos = self._aes.decrypt(crudo[:12], crudo[12:], self._aad(id_))
        except Exception:
            raise BaseDatosComprometidaError(f"El template de '{id_}' ha sido manipulado") from None
        t = np.frombuffer(datos, dtype="<f4").astype(np.float32)
        if t.shape != (TAM_TEMPLATE,):
            raise BaseDatosComprometidaError(f"El template de '{id_}' tiene un tamaño incorrecto")
        return t

    # ---------------------------------------------------------------- lectura
    def cargar(self) -> list[tuple[str, np.ndarray]]:
        with bloqueo(self.ruta_bloqueo):
            return self._cargar()

    def _cargar(self) -> list[tuple[str, np.ndarray]]:
        """Devuelve [(id, template), ...]. Lanza BaseDatosComprometidaError si algo no cuadra."""
        if not self.ruta.exists():
            if self._version_guardada() > 0:
                raise BaseDatosComprometidaError(
                    f"Falta {self.ruta.name} pero el contador dice que había datos: ¿la han borrado?")
            self._version = 0
            return []
        try:
            texto = self.ruta.read_bytes().decode("utf-8").replace("\r\n", "\n")
        except UnicodeDecodeError:
            raise BaseDatosComprometidaError(f"{self.ruta.name} no es texto UTF-8") from None
        cuerpo, sep, ultima = texto.rstrip("\n").rpartition("\n")
        if not sep or not ultima.startswith(MARCA_FIRMA + ","):
            raise BaseDatosComprometidaError(f"Falta la firma al final de {self.ruta.name}")
        valor = ultima[len(MARCA_FIRMA) + 1:]
        version_txt, _, firma = valor.partition(":")
        if not version_txt.isdigit() or not firma:
            raise BaseDatosComprometidaError("Línea de firma con formato incorrecto")
        version = int(version_txt)
        cuerpo_b = (cuerpo + "\n").encode("utf-8")
        if not hmac.compare_digest(firma, self._mac(cuerpo_b, version)):
            raise BaseDatosComprometidaError(
                f"La firma de {self.ruta.name} no coincide: el fichero ha sido modificado fuera del sistema")
        guardada = self._version_guardada()
        if version < guardada:
            raise BaseDatosComprometidaError(
                f"{self.ruta.name} es una versión antigua ({version} < {guardada}): posible rollback")
        if version > guardada:          # un corte justo antes de actualizar el contador
            self._guardar_version(version)
        self._version = version

        lector = csv.reader(io.StringIO(cuerpo + "\n"))
        if next(lector, None) != CABECERA:
            raise BaseDatosComprometidaError(f"Cabecera de {self.ruta.name} incorrecta")
        filas, vistos = [], set()
        for fila in lector:
            if len(fila) != 2:
                raise BaseDatosComprometidaError(f"Fila con formato incorrecto en {self.ruta.name}")
            id_, campo = fila
            if clave_id(id_) in vistos:
                raise BaseDatosComprometidaError(f"Identificador duplicado en {self.ruta.name}: '{id_}'")
            vistos.add(clave_id(id_))
            filas.append((id_, self._descifrar(id_, campo)))
        return filas

    # ---------------------------------------------------------------- escritura
    def transaccion(self):
        """Bloqueo para hacer comprobaciones + escritura sin que otro proceso se cuele."""
        return bloqueo(self.ruta_bloqueo)

    def anadir(self, id_: str, template: np.ndarray) -> None:
        with bloqueo(self.ruta_bloqueo):
            actuales = self._cargar()                       # verifica antes de escribir
            if any(clave_id(i) == clave_id(id_) for i, _ in actuales):
                raise ValueError(f"'{id_}' ya está en la base de datos")
            filas = [(i, self._cifrar(i, t)) for i, t in actuales]
            filas.append((id_, self._cifrar(id_, template)))
            self._escribir(filas)

    def _escribir(self, filas: list[tuple[str, str]]) -> None:
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(CABECERA)
        w.writerows(filas)
        cuerpo = buf.getvalue().encode("utf-8")
        version = max(self._version, self._version_guardada()) + 1
        firma = f"{MARCA_FIRMA},{version}:{self._mac(cuerpo, version)}\n".encode("ascii")
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        _escritura_atomica(self.ruta, cuerpo + firma)      # un único reemplazo atómico
        self._guardar_version(version)                     # si se corta aquí, se repara al leer
        self._version = version


def _escritura_atomica(ruta: Path, contenido: bytes) -> None:
    tmp = ruta.with_name(f".{ruta.name}.{secrets.token_hex(4)}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(contenido)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ruta)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
