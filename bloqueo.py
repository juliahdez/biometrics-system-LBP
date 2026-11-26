"""
Bloqueo exclusivo entre procesos sobre un fichero (Linux, macOS y Windows).

Sin esto, dos inscripciones a la vez se pisan: cada proceso escribe bd.csv y su
firma por separado, quedan desemparejados y la BD se da por manipulada para
siempre (encontrado atacando el sistema como lo haría el grupo rival).

Es reentrante dentro del mismo proceso: anadir() puede llamar a cargar() sin
bloquearse a sí mismo.
"""
from __future__ import annotations

import contextlib
import os
import threading
import time
from pathlib import Path

_local = threading.local()
ESPERA_MAX = 60.0   # segundos esperando a que otro proceso suelte el bloqueo


class BloqueoError(Exception):
    pass


@contextlib.contextmanager
def bloqueo(ruta: Path):
    ruta = Path(ruta).resolve()
    niveles = getattr(_local, "niveles", None)
    if niveles is None:
        niveles = _local.niveles = {}
    if niveles.get(ruta, 0) > 0:            # ya lo tenemos: reentrada
        niveles[ruta] += 1
        try:
            yield
        finally:
            niveles[ruta] -= 1
        return

    ruta.parent.mkdir(parents=True, exist_ok=True)
    f = open(ruta, "a+b")
    try:
        _adquirir(f)
        niveles[ruta] = 1
        try:
            yield
        finally:
            niveles[ruta] = 0
            _liberar(f)
    finally:
        f.close()


if os.name == "nt":
    import msvcrt

    def _adquirir(f):
        limite = time.monotonic() + ESPERA_MAX
        while True:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if time.monotonic() > limite:
                    raise BloqueoError("La base de datos está ocupada por otro proceso") from None
                time.sleep(0.05)

    def _liberar(f):
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
else:
    import fcntl

    def _adquirir(f):
        limite = time.monotonic() + ESPERA_MAX
        while True:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError:
                if time.monotonic() > limite:
                    raise BloqueoError("La base de datos está ocupada por otro proceso") from None
                time.sleep(0.05)

    def _liberar(f):
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
