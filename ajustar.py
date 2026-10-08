"""
Prueba varias combinaciones de parámetros del LBP y dice cuál da menos EER.

Usar con fotos REALES (una subcarpeta por persona, 2+ fotos por persona):
    python ajustar.py fotos/

Después, copiad los mejores PUNTOS / RADIO / REJILLA en lbp.py.
OJO: si ajustáis con las mismas fotos con las que luego evaluáis, el resultado
sale optimista. Lo correcto es ajustar con una parte y evaluar con otra.
"""
import argparse
from pathlib import Path

import numpy as np
from skimage.feature import local_binary_pattern

from detector import ErrorDetector, cargar_imagen, detectar
from evaluar import EXTENSIONES, curvas
from preprocesado import preprocesar

COMBINACIONES = [  # (PUNTOS, RADIO, REJILLA)
    (8, 1, 8), (8, 2, 8), (8, 2, 7), (8, 2, 6), (8, 2, 4),
    (8, 1, 4), (8, 3, 8), (16, 2, 8),
]


def template(cara, P, R, n_rej):
    n = P * (P - 1) + 3
    cod = local_binary_pattern(cara, P, R, "nri_uniform").astype(np.int64)
    lado = cara.shape[0] // n_rej
    hs = []
    for i in range(n_rej):
        for j in range(n_rej):
            h = np.bincount(cod[i * lado:(i + 1) * lado, j * lado:(j + 1) * lado].ravel(),
                            minlength=n).astype(np.float64)
            hs.append(h / h.sum())
    return np.concatenate(hs)


def matriz_chi2(T, n_rej):
    D = np.zeros((len(T), len(T)))
    for i in range(len(T)):
        s, q = T[i] + T, (T[i] - T) ** 2
        D[i] = np.divide(q, s, out=np.zeros_like(q), where=s > 0).sum(1) / n_rej ** 2
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    args = ap.parse_args()

    caras, etiquetas = [], []
    for persona in sorted(p for p in Path(args.carpeta).iterdir() if p.is_dir()):
        for foto in sorted(f for f in persona.iterdir() if f.suffix.lower() in EXTENSIONES):
            try:
                img = cargar_imagen(foto)
                caras.append(preprocesar(img, *detectar(img)))
                etiquetas.append(persona.name)
            except ErrorDetector:
                pass
    etiquetas = np.array(etiquetas)
    print(f"{len(caras)} caras de {len(set(etiquetas))} personas\n")

    iu = np.triu_indices(len(caras), 1)
    misma = (etiquetas[:, None] == etiquetas[None, :])[iu]
    resultados = []
    for P, R, n_rej in COMBINACIONES:
        D = matriz_chi2(np.array([template(c, P, R, n_rej) for c in caras]), n_rej)
        umbrales, far, frr, k = curvas(D[iu][misma], D[iu][~misma])
        np.fill_diagonal(D, np.inf)
        rank1 = (etiquetas[D.argmin(1)] == etiquetas).mean()
        eer = (far[k] + frr[k]) / 2
        resultados.append((eer, rank1, P, R, n_rej, umbrales[k]))
        print(f"P={P:2d} R={R} rejilla={n_rej}x{n_rej}:  EER {100 * eer:5.2f} %   "
              f"rank-1 {100 * rank1:5.1f} %   umbral EER {umbrales[k]:.4f}")

    mejor = min(resultados)
    print(f"\nMejor: PUNTOS={mejor[2]}, RADIO={mejor[3]}, REJILLA=({mejor[4]}, {mejor[4]})"
          f"  ->  EER {100 * mejor[0]:.2f} %")


if __name__ == "__main__":
    main()
