"""
Pasa el detector por todas las fotos de una carpeta (por ejemplo, la base de datos del profe).

Uso:
    python probar_carpeta.py carpeta_fotos/
    python probar_carpeta.py carpeta_fotos/ --salida resultados/

Genera en la carpeta de salida:
    - informe.csv           una fila por foto: estado, nº de caras, confianza, caja, ojos, ángulo
    - ok/                   fotos con la cara detectada (caja y puntos dibujados)
    - varias/               fotos con más de una cara (verde = la elegida)
    - sin_cara/             fotos donde no se encontró cara
    - error/                ficheros que no se pudieron leer
"""
import argparse
import csv
import shutil
from pathlib import Path

import cv2

from detector import DetectorCaras, ErrorDetector, cargar_imagen, dibujar, guardar_imagen

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pgm", ".ppm"}


def main():
    ap = argparse.ArgumentParser(description="Prueba el detector sobre una carpeta de fotos")
    ap.add_argument("carpeta", help="carpeta con las fotos (se buscan también en subcarpetas)")
    ap.add_argument("--salida", default="resultados_detector", help="carpeta de resultados")
    ap.add_argument("--umbral", type=float, default=0.8, help="confianza mínima (por defecto 0.8)")
    args = ap.parse_args()

    entrada = Path(args.carpeta)
    fotos = sorted(p for p in entrada.rglob("*") if p.suffix.lower() in EXTENSIONES)
    if not fotos:
        raise SystemExit(f"No hay fotos en {entrada}")

    salida = Path(args.salida)
    if salida.exists():
        shutil.rmtree(salida)
    for sub in ("ok", "varias", "sin_cara", "error"):
        (salida / sub).mkdir(parents=True)

    det = DetectorCaras(umbral_confianza=args.umbral)
    cuenta = {"ok": 0, "varias": 0, "sin_cara": 0, "error": 0}
    filas = []

    for i, foto in enumerate(fotos, 1):
        nombre = str(foto.relative_to(entrada))
        nombre_plano = nombre.replace("/", "__").replace("\\", "__")
        nombre_jpg = str(Path(nombre_plano).with_suffix(".jpg"))
        fila = {"foto": nombre, "estado": "", "n_caras": 0, "confianza": "",
                "caja": "", "ojo_izq": "", "ojo_der": "", "angulo_ojos": "", "detalle": ""}
        try:
            img = cargar_imagen(foto)
            caras = det.detectar_todas(img)
            fila["n_caras"] = len(caras)
            if not caras:
                estado = "sin_cara"
                guardar_imagen(salida / estado / nombre_jpg, img)
            else:
                estado = "ok" if len(caras) == 1 else "varias"
                c = caras[0]
                fila.update(confianza=f"{c.confianza:.3f}", caja=c.caja,
                            ojo_izq=tuple(round(v) for v in c.ojo_izq),
                            ojo_der=tuple(round(v) for v in c.ojo_der),
                            angulo_ojos=f"{c.angulo_ojos:.1f}")
                guardar_imagen(salida / estado / nombre_jpg, dibujar(img, caras))
        except ErrorDetector as e:
            estado = "error"
            fila["detalle"] = str(e)
            shutil.copy(foto, salida / estado / nombre_plano)

        fila["estado"] = estado
        cuenta[estado] += 1
        filas.append(fila)
        print(f"[{i}/{len(fotos)}] {estado:8s} {nombre}")

    with open(salida / "informe.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    total = len(fotos)
    print("\n===== RESUMEN =====")
    for k, v in cuenta.items():
        print(f"{k:9s} {v:4d}  ({100 * v / total:.1f}%)")
    confs = [float(r["confianza"]) for r in filas if r["confianza"]]
    if confs:
        print(f"confianza media {sum(confs) / len(confs):.3f}, mínima {min(confs):.3f}")
    print(f"\nRevisad a mano las carpetas de {salida}/ y el informe.csv")


if __name__ == "__main__":
    main()
