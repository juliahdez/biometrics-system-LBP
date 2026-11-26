"""
Evaluación del rendimiento del sistema sobre una carpeta de fotos.

Estructura: una subcarpeta por persona, con 2 o más fotos cada una.
    fotos/
      persona_A/  1.jpg  2.jpg ...
      persona_B/  ...

Uso:
    python evaluar.py fotos/ --salida resultados_eval/

Qué mide:
  1. Todos contra todos: distancias genuinas (misma persona) e impostoras,
     FAR / FRR para cada umbral, EER y umbrales recomendados.
  2. Protocolo de la práctica, con el sistema real (BD .csv cifrada temporal):
     se inscribe la 1ª foto de cada persona y con el resto se hace
       - verificar(id_propio)   -> tasa de falsos rechazos (FRR)
       - verificar(id_ajeno)    -> tasa de falsas aceptaciones (FAR)
       - identificar()          -> aciertos, errores y no identificados
     con el umbral de config.py. Lo mismo para la vía sin detector (_sd).
Genera informe.md, distancias.csv, curvas.csv y curvas.png.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import tempfile
from pathlib import Path

import numpy as np

import config
from detector import ErrorDetector, cargar_imagen, detectar
from lbp import comparar
from pipeline import obtener_template
from sistema import SistemaBiometrico

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".pgm", ".ppm"}


def listar_fotos(carpeta: Path):
    return {p.name: sorted(f for f in p.iterdir() if f.suffix.lower() in EXTENSIONES)
            for p in sorted(carpeta.iterdir()) if p.is_dir()}


def curvas(genuinas, impostoras, n=1000):
    umbrales = np.linspace(0, max(genuinas.max(), impostoras.max()) * 1.01, n)
    far = np.array([(impostoras <= u).mean() for u in umbrales])
    frr = np.array([(genuinas > u).mean() for u in umbrales])
    i = int(np.argmin(np.abs(far - frr)))
    return umbrales, far, frr, i


def protocolo(fotos: dict, sin_detector: bool):
    """Inscribe la 1ª foto de cada persona en una BD temporal y prueba el resto."""
    with tempfile.TemporaryDirectory() as tmp:
        s = SistemaBiometrico(ruta_bd=Path(tmp) / "bd.csv", ruta_clave=Path(tmp) / "clave.key",
                              limitar_intentos=False)   # se miden cientos de intentos por id
        prep = {}
        for persona, lista in fotos.items():
            for f in lista:
                img = cargar_imagen(f)
                if sin_detector:   # se simula un atacante que da la cara ya recortada
                    try:
                        (x, y, w, h), _, _ = detectar(img)
                    except ErrorDetector:
                        continue
                    img = img[y:y + h, x:x + w]
                prep.setdefault(persona, []).append(img)
        inscritos, fallos_insc = [], []
        for persona, imgs in prep.items():
            try:
                (s.inscribir_sd if sin_detector else s.inscribir)(persona, imgs[0])
                inscritos.append(persona)
            except Exception as e:  # noqa: BLE001
                fallos_insc.append((persona, str(e)))
        ver = s.verificar_sd if sin_detector else s.verificar
        ide = s.identificar_sd if sin_detector else s.identificar
        g_ok = g_tot = i_ok = i_tot = 0
        id_bien = id_mal = id_nadie = 0
        for persona in inscritos:
            for img in prep[persona][1:]:
                g_tot += 1
                g_ok += ver(persona, img)
                for otra in inscritos:
                    if otra != persona:
                        i_tot += 1
                        i_ok += ver(otra, img)
                r = ide(img)
                if r == persona:
                    id_bien += 1
                elif r is None:
                    id_nadie += 1
                else:
                    id_mal += 1
        return dict(inscritos=len(inscritos), fallos_insc=fallos_insc,
                    frr=1 - g_ok / max(g_tot, 1), n_gen=g_tot,
                    far=i_ok / max(i_tot, 1), n_imp=i_tot,
                    id_bien=id_bien, id_mal=id_mal, id_nadie=id_nadie, n_id=id_bien + id_mal + id_nadie)


def no_inscritos(templates_por_persona: dict):
    """Identificación de conjunto abierto: una persona NO inscrita intenta identificarse.

    Para cada persona P se inscribe la 1ª foto de todas las demás y se identifican
    todas las fotos de P. Lo correcto es "no identificado"; cualquier nombre es un error.
    """
    personas = [p for p, ts in templates_por_persona.items() if ts]
    con_nombre = total = 0
    for fuera in personas:
        with tempfile.TemporaryDirectory() as tmp:
            s = SistemaBiometrico(ruta_bd=Path(tmp) / "bd.csv", ruta_clave=Path(tmp) / "clave.key",
                                  limitar_intentos=False)
            for p in personas:
                if p != fuera:
                    s.inscribir_t(p, templates_por_persona[p][0])
            for t in templates_por_persona[fuera]:
                total += 1
                con_nombre += s.identificar_t(t) is not None
    return con_nombre, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--salida", default="resultados_eval")
    a = ap.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    fotos = listar_fotos(Path(a.carpeta))
    n_fotos = sum(len(v) for v in fotos.values())

    # ---------------------------------------------------------------- 1) todos contra todos
    templates, fallos = [], []
    for persona, lista in fotos.items():
        for f in lista:
            try:
                templates.append((persona, f.name, obtener_template(f)))
            except ErrorDetector as e:
                fallos.append((persona, f.name, str(e)))
    genuinas, impostoras, filas = [], [], []
    for (p1, f1, t1), (p2, f2, t2) in itertools.combinations(templates, 2):
        d = comparar(t1, t2)
        (genuinas if p1 == p2 else impostoras).append(d)
        filas.append((p1, f1, p2, f2, "genuina" if p1 == p2 else "impostora", f"{d:.5f}"))
    if not genuinas or not impostoras:
        raise SystemExit("Hacen falta al menos 2 personas y alguna con 2 o más fotos")
    genuinas, impostoras = np.array(genuinas), np.array(impostoras)
    with open(salida / "distancias.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["persona1", "foto1", "persona2", "foto2", "tipo", "distancia"])
        w.writerows(filas)
    umbrales, far, frr, i_eer = curvas(genuinas, impostoras)
    with open(salida / "curvas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["umbral", "FAR", "FRR"])
        w.writerows(zip(umbrales.round(5), far.round(5), frr.round(5)))
    eer = (far[i_eer] + frr[i_eer]) / 2

    def punto_far(objetivo):
        idx = np.where(far <= objetivo)[0]
        k = idx[-1] if len(idx) else 0
        return umbrales[k], far[k], frr[k]

    def en(u):
        return (impostoras <= u).mean(), (genuinas > u).mean()

    # ---------------------------------------------------------------- 2) protocolo con el sistema
    print("Protocolo con detector...")
    p_det = protocolo(fotos, sin_detector=False)
    print("Protocolo sin detector (_sd)...")
    p_sd = protocolo(fotos, sin_detector=True)
    print("Personas no inscritas...")
    por_persona = {}
    for p, _, t in templates:
        por_persona.setdefault(p, []).append(t)
    abiertos, n_abiertos = no_inscritos(por_persona)

    # ---------------------------------------------------------------- informe
    L = []
    L.append("# Evaluación de rendimiento\n")
    L.append(f"- Fotos: {n_fotos} de {len(fotos)} personas")
    L.append(f"- Fallos del detector: {len(fallos)} ({100 * len(fallos) / n_fotos:.1f} %)")
    for p, f, e in fallos:
        L.append(f"  - {p}/{f}: {e}")
    L.append(f"- Umbral del sistema (config.py): **{config.UMBRAL}**\n")
    L.append("## 1. Todos contra todos\n")
    L.append(f"- Pares genuinos: {len(genuinas)} (distancia media {genuinas.mean():.3f}, máx {genuinas.max():.3f})")
    L.append(f"- Pares impostores: {len(impostoras)} (distancia media {impostoras.mean():.3f}, mín {impostoras.min():.3f})")
    L.append(f"- **EER = {100 * eer:.2f} %** con umbral {umbrales[i_eer]:.4f}\n")
    L.append("| Punto de trabajo | Umbral | FAR | FRR |")
    L.append("|---|---|---|---|")
    for nombre, obj in (("FAR ≤ 1 %", 0.01), ("FAR ≤ 5 %", 0.05), ("FAR ≤ 10 %", 0.10)):
        u, fa, fr = punto_far(obj)
        L.append(f"| {nombre} | {u:.4f} | {100 * fa:.2f} % | {100 * fr:.2f} % |")
    L.append(f"| EER | {umbrales[i_eer]:.4f} | {100 * far[i_eer]:.2f} % | {100 * frr[i_eer]:.2f} % |")
    fa, fr = en(config.UMBRAL)
    L.append(f"| **Umbral del sistema** | {config.UMBRAL:.4f} | {100 * fa:.2f} % | {100 * fr:.2f} % |\n")
    L.append("## 2. Protocolo de la práctica (1 foto inscrita por persona, el resto como intentos)\n")
    L.append("| Vía | Inscritos | FRR (verificar propio) | FAR (verificar ajeno) | Identificación correcta | Identificado como otro | No identificado |")
    L.append("|---|---|---|---|---|---|---|")
    for nombre, p in (("Con detector", p_det), ("Sin detector (_sd)", p_sd)):
        n = max(p["n_id"], 1)
        L.append(f"| {nombre} | {p['inscritos']} | {100 * p['frr']:.1f} % ({p['n_gen']}) | "
                 f"{100 * p['far']:.1f} % ({p['n_imp']}) | {100 * p['id_bien'] / n:.1f} % | "
                 f"{100 * p['id_mal'] / n:.1f} % | {100 * p['id_nadie'] / n:.1f} % |")
    for nombre, p in (("con detector", p_det), ("sin detector", p_sd)):
        for persona, e in p["fallos_insc"]:
            L.append(f"\n- Inscripción rechazada ({nombre}): {persona}: {e}")
    L.append("\n## 3. Persona NO inscrita que intenta identificarse (conjunto abierto)\n")
    L.append(f"Se inscribe a todos menos a una persona y se identifican sus fotos (repetido con cada persona). "
             f"Lo correcto es \"no identificado\".\n")
    L.append(f"- **Le da el nombre de otra persona: {abiertos}/{n_abiertos} "
             f"({100 * abiertos / max(n_abiertos, 1):.1f} %)**")
    L.append(f"- Con {len(por_persona) - 1} inscritos. Con 51 inscritos será mayor: hay más candidatos con los que "
             f"confundirse.")
    texto = "\n".join(L) + "\n"
    (salida / "informe.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        bins = np.linspace(0, umbrales[-1], 50)
        a1.hist(genuinas, bins, alpha=0.6, density=True, label="genuinas (misma persona)")
        a1.hist(impostoras, bins, alpha=0.6, density=True, label="impostoras")
        a1.axvline(config.UMBRAL, color="k", ls="--", label=f"umbral ({config.UMBRAL})")
        a1.set_xlabel("distancia chi-cuadrado"); a1.legend(fontsize=8); a1.set_title("Distribución de distancias")
        a2.plot(umbrales, far, label="FAR"); a2.plot(umbrales, frr, label="FRR")
        a2.axvline(config.UMBRAL, color="k", ls="--")
        a2.set_xlabel("umbral"); a2.legend(); a2.set_title(f"FAR / FRR (EER {100 * eer:.1f} %)")
        fig.tight_layout(); fig.savefig(salida / "curvas.png", dpi=120)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
