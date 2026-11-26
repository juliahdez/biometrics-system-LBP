"""
Línea de comandos del sistema biométrico.

    python biometria.py inscribir      ID FOTO
    python biometria.py verificar      ID FOTO
    python biometria.py identificar    FOTO
    python biometria.py inscribir_sd   ID FOTO        (sin detector: FOTO = cara ya recortada)
    python biometria.py verificar_sd   ID FOTO
    python biometria.py identificar_sd FOTO
    python biometria.py inscribir_t    ID TEMPLATE    (TEMPLATE = .npy / .csv / .txt)
    python biometria.py verificar_t    ID TEMPLATE
    python biometria.py identificar_t  TEMPLATE

    python biometria.py extraer FOTO SALIDA.npy [--sd]   saca el template de una foto
    python biometria.py inscribir_carpeta CARPETA        inscribe a todos (ver README)
    python biometria.py listar                           ids inscritos

Códigos de salida: 0 = aceptado / hecho, 1 = rechazado / no identificado, 2 = error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

import sistema as sb

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pgm", ".ppm"}


def inscribir_carpeta(s: sb.SistemaBiometrico, carpeta: Path) -> int:
    """
    Dos formatos:
      carpeta/Nombre Apellido.jpg            -> id = nombre del fichero
      carpeta/Nombre Apellido/foto.jpg       -> id = nombre de la subcarpeta (1ª foto)
    """
    entradas = []
    for p in sorted(carpeta.iterdir()):
        if p.is_dir():
            fotos = sorted(f for f in p.iterdir() if f.suffix.lower() in EXTENSIONES)
            if fotos:
                entradas.append((p.name, fotos[0]))
        elif p.suffix.lower() in EXTENSIONES:
            entradas.append((p.stem, p))
    ok = 0
    for id_, foto in entradas:
        try:
            s.inscribir(id_, foto)
            print(f"  OK     {id_}")
            ok += 1
        except sb.YaInscritoError:
            print(f"  YA     {id_} (ya estaba inscrito)")
        except Exception as e:  # noqa: BLE001 - se informa y se sigue con el resto
            print(f"  FALLO  {id_}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(entradas)} inscritos ahora. Total en la base de datos: {len(s.listar())}")
    return 0 if ok == len(entradas) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="biometria", description="Sistema de reconocimiento facial LBP")
    # Sin opciones para cambiar el umbral, la BD o la clave, ni para ver la distancia:
    # con --umbral 5 se aceptaba a cualquiera, con --bd/--clave se usaba una BD fabricada
    # por el atacante y la distancia permite ataques de hill-climbing. Las rutas son
    # configuración del despliegue (variables de entorno, ver config.py).
    ap.add_argument("-v", "--verbose", action="store_true", help="mostrar avisos internos (sin distancias)")
    sub = ap.add_subparsers(dest="orden", required=True)

    for nombre in ("inscribir", "verificar", "inscribir_sd", "verificar_sd"):
        p = sub.add_parser(nombre)
        p.add_argument("id")
        p.add_argument("foto")
    for nombre in ("identificar", "identificar_sd"):
        sub.add_parser(nombre).add_argument("foto")
    for nombre in ("inscribir_t", "verificar_t"):
        p = sub.add_parser(nombre)
        p.add_argument("id")
        p.add_argument("template")
    sub.add_parser("identificar_t").add_argument("template")
    p = sub.add_parser("extraer")
    p.add_argument("foto")
    p.add_argument("salida")
    p.add_argument("--sd", action="store_true", help="sin detector")
    sub.add_parser("inscribir_carpeta").add_argument("carpeta")
    sub.add_parser("listar")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="[%(levelname)s] %(message)s")

    try:
        if a.orden == "extraer":
            t = sb.extraer_template(a.foto, sin_detector=a.sd)
            np.save(a.salida, t)
            print(f"Template guardado en {a.salida} ({t.size} valores)")
            return 0

        s = sb.SistemaBiometrico()

        if a.orden == "listar":
            ids = s.listar()
            print("\n".join(ids) if ids else "(base de datos vacía)")
            print(f"\nTotal: {len(ids)}")
            return 0
        if a.orden == "inscribir_carpeta":
            return inscribir_carpeta(s, Path(a.carpeta))

        metodo = getattr(s, a.orden)
        if a.orden.startswith("inscribir"):
            metodo(a.id, a.foto if hasattr(a, "foto") else a.template)
            print(f"INSCRITO: {a.id}")
            return 0
        if a.orden.startswith("verificar"):
            ok = metodo(a.id, a.foto if hasattr(a, "foto") else a.template)
            print("ACEPTADO" if ok else "RECHAZADO")
            return 0 if ok else 1
        id_ = metodo(a.foto if hasattr(a, "foto") else a.template)
        print(f"IDENTIFICADO: {id_}" if id_ else "NO IDENTIFICADO")
        return 0 if id_ else 1

    except (sb.ErrorBiometrico, sb.ErrorDetector, sb.ErrorTemplate,
            sb.BaseDatosComprometidaError, sb.ClaveError, sb.BloqueoError) as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 2
    except PermissionError as e:
        print(f"ERROR: sin permiso para acceder a {e.filename}. En Docker sobre Linux, "
              f'añade --user "$(id -u):$(id -g)" (ver README).', file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - nunca un traceback hacia fuera
        print(f"ERROR inesperado ({type(e).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
