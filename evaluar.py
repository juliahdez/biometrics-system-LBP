"""
Evaluación del sistema (detector + preprocesado + LBP) sobre una carpeta de fotos.

Estructura esperada: una subcarpeta por persona
    fotos/
      persona_A/  foto1.jpg  foto2.jpg ...
      persona_B/  ...

Uso:
    python evaluar.py fotos/ --salida resultados_eval/

Calcula todas las distancias genuinas (misma persona) e impostoras (personas
distintas) y saca:
    - FAR / FRR para cada umbral, EER y umbral recomendado
    - tasa de acierto en identificación (rank-1, dejando una foto fuera)
    - distancias.csv, curvas.csv y una gráfica (si hay matplotlib)
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np

from detector import ErrorDetector
from lbp import comparar
from pipeline import obtener_template

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pgm", ".ppm"}


def cargar_templates(carpeta: Path):
    templates, fallos = [], []
    for persona in sorted(p for p in carpeta.iterdir() if p.is_dir()):
        for foto in sorted(f for f in persona.iterdir() if f.suffix.lower() in EXTENSIONES):
            try:
                templates.append((persona.name, foto.name, obtener_template(foto)))
            except ErrorDetector as e:
                fallos.append((persona.name, foto.name, str(e)))
    return templates, fallos


def curvas(genuinas, impostoras, n=500):
    umbrales = np.linspace(0, max(genuinas.max(), impostoras.max()) * 1.01, n)
    far = np.array([(impostoras <= u).mean() for u in umbrales])   # impostores aceptados
    frr = np.array([(genuinas > u).mean() for u in umbrales])      # genuinos rechazados
    i = int(np.argmin(np.abs(far - frr)))
    return umbrales, far, frr, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--salida", default="resultados_eval")
    args = ap.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    templates, fallos = cargar_templates(Path(args.carpeta))
    print(f"{len(templates)} templates, {len(fallos)} fotos sin cara o erróneas")
    for p, f, e in fallos:
        print(f"   FALLO {p}/{f}: {e}")

    genuinas, impostoras, filas = [], [], []
    for (p1, f1, t1), (p2, f2, t2) in itertools.combinations(templates, 2):
        d = comparar(t1, t2)
        tipo = "genuina" if p1 == p2 else "impostora"
        (genuinas if p1 == p2 else impostoras).append(d)
        filas.append((p1, f1, p2, f2, tipo, f"{d:.5f}"))
    if not genuinas or not impostoras:
        raise SystemExit("Hacen falta al menos 2 personas y alguna persona con 2+ fotos")
    genuinas, impostoras = np.array(genuinas), np.array(impostoras)

    with open(salida / "distancias.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona1", "foto1", "persona2", "foto2", "tipo", "distancia"])
        w.writerows(filas)

    umbrales, far, frr, i = curvas(genuinas, impostoras)
    with open(salida / "curvas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["umbral", "FAR", "FRR"])
        w.writerows(zip(umbrales.round(5), far.round(5), frr.round(5)))

    # Umbral más permisivo con FAR <= 1 %
    seguros = np.where(far <= 0.01)[0]
    i_seg = seguros[-1] if len(seguros) else 0

    # Identificación rank-1: cada template contra todos los demás
    aciertos = total = 0
    for k, (p, _, t) in enumerate(templates):
        if sum(1 for q, _, _ in templates if q == p) < 2:
            continue                                 # su persona no tiene otra foto
        dist = [(comparar(t, t2), p2) for j, (p2, _, t2) in enumerate(templates) if j != k]
        aciertos += min(dist)[1] == p
        total += 1

    print("\n===== RESULTADOS =====")
    print(f"Pares genuinos:   {len(genuinas):6d}  distancia media {genuinas.mean():.3f} (máx {genuinas.max():.3f})")
    print(f"Pares impostores: {len(impostoras):6d}  distancia media {impostoras.mean():.3f} (mín {impostoras.min():.3f})")
    print(f"EER = {100 * (far[i] + frr[i]) / 2:.2f} %  con umbral {umbrales[i]:.4f}")
    print(f"Umbral con FAR <= 1 %: {umbrales[i_seg]:.4f}  ->  FRR = {100 * frr[i_seg]:.2f} %")
    if total:
        print(f"Identificación rank-1: {aciertos}/{total} = {100 * aciertos / total:.1f} %")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        bins = np.linspace(0, umbrales[-1], 60)
        a1.hist(genuinas, bins, alpha=0.6, density=True, label="genuinas")
        a1.hist(impostoras, bins, alpha=0.6, density=True, label="impostoras")
        a1.axvline(umbrales[i], color="k", ls="--", label=f"EER ({umbrales[i]:.3f})")
        a1.set_xlabel("distancia chi-cuadrado"); a1.legend(); a1.set_title("Distribución de distancias")
        a2.plot(umbrales, far, label="FAR"); a2.plot(umbrales, frr, label="FRR")
        a2.axvline(umbrales[i], color="k", ls="--")
        a2.set_xlabel("umbral"); a2.legend(); a2.set_title("FAR / FRR")
        fig.tight_layout(); fig.savefig(salida / "curvas.png", dpi=120)
        print(f"Gráfica en {salida / 'curvas.png'}")
    except ImportError:
        print("(instala matplotlib para la gráfica)")


if __name__ == "__main__":
    main()
