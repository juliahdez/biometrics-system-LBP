"""
Evaluación de seguridad: ejecuta los ataques que pide el enunciado contra una copia
temporal del sistema y genera un informe (informe_ataques.md).

    python ataques.py fotos/ --salida resultados/ataques/

`fotos/` tiene una subcarpeta por persona (como en evaluar.py). Se inscribe la 1ª
foto de cada persona y el resto se usan como intentos. No toca la base de datos real.

Escenarios:
  A. Ataques de presentación (por la entrada normal, con detector)
  B. Saltarse el detector (métodos _sd)
  C. Saltarse detector y extractor (métodos _t)
  D. Leer la base de datos
  E. Escribir en la base de datos
  F. Entradas maliciosas (ids, ficheros)
  G. Encontrados atacando el sistema como lo haría el grupo rival (y ya corregidos)
  H. Encontrados por una revisión externa (y ya corregidos o documentados)

Cada resultado se marca como:
  DEFENDIDO              el ataque falla
  LIMITACIÓN CONOCIDA    el ataque funciona (o funciona en parte) y está documentado
"""
from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import cv2
import numpy as np

import config
from basedatos import BaseDatosComprometidaError
from detector import ErrorDetector, cargar_imagen, detectar
from evaluar import listar_fotos
from lbp import N_CASILLAS, N_CELDAS, TAM_TEMPLATE, ErrorTemplate, distancias
from sistema import ErrorBiometrico, SistemaBiometrico, extraer_template

DEF, LIM = "DEFENDIDO", "LIMITACIÓN CONOCIDA"


class Informe:
    def __init__(self):
        self.filas = []

    def add(self, grupo, ataque, resultado, estado, nota=""):
        self.filas.append((grupo, ataque, resultado, estado, nota))
        print(f"[{estado:19s}] {grupo} | {ataque}: {resultado}")

    def markdown(self, cabecera: str) -> str:
        L = [cabecera, "", "| # | Ataque | Resultado | Estado | Comentario |", "|---|---|---|---|---|"]
        grupo_actual = None
        n = 0
        for grupo, ataque, res, est, nota in self.filas:
            if grupo != grupo_actual:
                L.append(f"| | **{grupo}** | | | |")
                grupo_actual = grupo
            n += 1
            icono = "✅" if est == DEF else "⚠️"
            L.append(f"| {n} | {ataque} | {res} | {icono} {est} | {nota} |")
        d = sum(1 for f in self.filas if f[3] == DEF)
        L += ["", f"**{d} de {len(self.filas)} ataques defendidos; {len(self.filas) - d} son limitaciones "
                  f"conocidas y documentadas.**"]
        return "\n".join(L) + "\n"


def intenta(f) -> tuple[bool, str]:
    """(lanzó excepción controlada, nombre) — para ataques que deben producir error."""
    try:
        r = f()
        return False, f"devolvió {r!r}"
    except (ErrorBiometrico, ErrorDetector, ErrorTemplate, BaseDatosComprometidaError) as e:
        return True, type(e).__name__
    except Exception as e:  # noqa: BLE001 - un error no controlado es un fallo del sistema
        return False, f"ERROR NO CONTROLADO {type(e).__name__}: {e}"


def plantillas_aleatorias(n, rng):
    return [rng.dirichlet(np.full(N_CASILLAS, 0.5), size=N_CELDAS).ravel().astype(np.float32)
            for _ in range(n)]


def _inscribir_en_paralelo(args):
    ruta, i = args
    s = SistemaBiometrico(ruta_bd=ruta / "bd.csv", ruta_clave=ruta / "clave.key")
    for k, t in enumerate(plantillas_aleatorias(6, np.random.default_rng(1000 + i))):
        s.inscribir_t(f"proceso {i} n {k}", t)


class config_temporal:
    def __init__(self, **valores):
        self.valores, self.previos = valores, {}

    def __enter__(self):
        for k, v in self.valores.items():
            self.previos[k] = getattr(config, k)
            setattr(config, k, v)

    def __exit__(self, *exc):
        for k, v in self.previos.items():
            setattr(config, k, v)


def png_bomba() -> bytes:
    """PNG de pocos bytes que dice medir 60000 x 60000 píxeles (3,6 Gpx)."""
    def trozo(tipo, datos):
        return struct.pack(">I", len(datos)) + tipo + datos + struct.pack(">I", zlib.crc32(tipo + datos))
    ihdr = struct.pack(">IIBBBBB", 60000, 60000, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + trozo(b"IHDR", ihdr)
            + trozo(b"IDAT", zlib.compress(b"\x00" * 60001 * 50)) + trozo(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("--salida", default="resultados_ataques")
    a = ap.parse_args()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    inf = Informe()

    fotos = listar_fotos(Path(a.carpeta))
    tmp = Path(tempfile.mkdtemp())
    try:
        # Sin límite de intentos para poder MEDIR (cientos de intentos por id); el límite
        # se evalúa aparte en el grupo G con un sistema que sí lo tiene activado.
        s = SistemaBiometrico(ruta_bd=tmp / "bd.csv", ruta_clave=tmp / "clave.key", limitar_intentos=False)
        inscritos, imgs = [], {}
        for persona, lista in fotos.items():
            imgs[persona] = [cargar_imagen(f) for f in lista]
            try:
                s.inscribir(persona, imgs[persona][0])
                inscritos.append(persona)
            except Exception as e:  # noqa: BLE001
                print(f"No se pudo inscribir {persona}: {e}")
        print(f"\n{len(inscritos)} personas inscritas en una BD temporal\n")
        victima = inscritos[0]
        foto_victima = imgs[victima][0]
        t_inscritos = np.stack([t for _, t in s.bd.cargar()])

        # ================================================================ A
        G = "A. Presentación (entrada normal)"
        # A1 impostor de esfuerzo cero
        acc = tot = 0
        for p in inscritos:
            for img in imgs[p][1:]:
                for otra in inscritos:
                    if otra != p:
                        tot += 1
                        acc += s.verificar(otra, img)
        inf.add(G, "Impostor de esfuerzo cero (foto de otra persona)",
                f"aceptado {acc}/{tot} ({100 * acc / tot:.1f} %)", LIM,
                "Es la FAR del sistema con el umbral elegido (punto EER). LBP no separa mejor "
                "caras no controladas; bajar el umbral reduce la FAR a costa de la FRR.")
        # A2 repetir la foto de inscripción (las 50 fotos son públicas para todos los grupos)
        ok = sum(s.verificar(p, imgs[p][0]) for p in inscritos)
        inf.add(G, "Presentar la misma foto con la que se inscribió (fotos públicas)",
                f"aceptado {ok}/{len(inscritos)}", LIM,
                "Las 50 fotos del profesor las tienen todos los grupos. Sin detección de vida "
                "(liveness) es imposible distinguir la foto de la persona.")
        # A3 la foto de inscripción retocada (impresa y escaneada, recomprimida...)
        def retoca(img):
            img = cv2.GaussianBlur(img, (0, 0), 1.2)
            img = np.clip(img.astype(np.float32) * 0.85 + 20, 0, 255).astype(np.uint8)
            img = cv2.resize(img, None, fx=0.6, fy=0.6)
            return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], 1)
        ok = sum(s.verificar(p, retoca(imgs[p][0])) for p in inscritos)
        inf.add(G, "Foto de inscripción degradada (desenfoque, luz, escala, JPEG 40)",
                f"aceptado {ok}/{len(inscritos)}", LIM, "Igual que A2: una copia de la foto sirve.")
        # A4 espejo
        ok = sum(s.verificar(p, cv2.flip(imgs[p][0], 1)) for p in inscritos)
        inf.add(G, "Foto de inscripción en espejo", f"aceptado {ok}/{len(inscritos)}", LIM,
                "El template es simétrico a propósito (mejora el reconocimiento).")
        # A5 foto sin cara / varias caras
        cafe = cv2.cvtColor(__import__("skimage").data.coffee(), cv2.COLOR_RGB2BGR)
        lanzo, nom = intenta(lambda: s.verificar(victima, cafe))
        inf.add(G, "Foto sin ninguna cara", nom, DEF if lanzo else LIM)
        h = foto_victima.shape[0]
        otra_img = imgs[inscritos[1]][1]
        otra_img = cv2.resize(otra_img, (int(otra_img.shape[1] * h / otra_img.shape[0] * 0.5), h // 2))
        lienzo = np.zeros((h, foto_victima.shape[1] + otra_img.shape[1], 3), np.uint8)
        lienzo[:, :foto_victima.shape[1]] = foto_victima
        lienzo[:otra_img.shape[0], foto_victima.shape[1]:] = otra_img
        ok = s.verificar(victima, lienzo)
        inf.add(G, "Foto con varias caras (víctima grande + atacante pequeño)",
                "aceptado" if ok else "rechazado", LIM if ok else DEF,
                "Se usa la cara más grande; para inscribir con más rigor: detectar_cara(varias='error').")

        # ================================================================ B
        G = "B. Saltarse el detector (_sd)"
        ok = tot = 0
        texturas = [rng.integers(0, 256, (160, 120, 3), dtype=np.uint8) for _ in range(30)]
        texturas += [cv2.cvtColor(__import__("skimage").data.__getattribute__(n)(), cv2.COLOR_RGB2BGR)
                     if __import__("skimage").data.__getattribute__(n)().ndim == 3 else
                     cv2.cvtColor(__import__("skimage").data.__getattribute__(n)(), cv2.COLOR_GRAY2BGR)
                     for n in ("coffee", "chelsea", "rocket", "brick", "grass", "gravel", "text", "moon")]
        acc_v = acc_i = 0
        for img in texturas:
            acc_i += s.identificar_sd(img) is not None
            for p in inscritos:
                tot += 1
                acc_v += s.verificar_sd(p, img)
        inf.add(G, f"Imágenes que no son caras ({len(texturas)}: ruido, gato, cohete, ladrillo...)",
                f"verificar_sd aceptó {acc_v}/{tot}; identificar_sd dio nombre {acc_i}/{len(texturas)}",
                DEF if acc_v / tot < 0.02 and acc_i == 0 else LIM,
                "Sin detector no se comprueba que haya una cara; protege el umbral y la regla anti-lobo.")
        lanzo, nom = intenta(lambda: s.verificar_sd(victima, np.full((200, 200, 3), 128, np.uint8)))
        inf.add(G, "Imagen lisa (sin contenido)", nom, DEF if lanzo else LIM)
        ok = 0
        for p in inscritos:
            (x, y, w, hh), _, _ = detectar(imgs[p][0])
            ok += s.verificar_sd(p, imgs[p][0][y:y + hh, x:x + w])
        inf.add(G, "Cara de inscripción recortada por la caja, presentada sin detector",
                f"aceptado {ok}/{len(inscritos)}", LIM,
                "Es la misma foto (A2) por otra vía. La vía _sd alinea con la posición media de los ojos, "
                "así que es coherente con la vía normal.")
        ok = sum(s.verificar_sd(p, imgs[p][0]) for p in inscritos)
        inf.add(G, "Foto completa (sin recortar) por la vía _sd",
                f"aceptado {ok}/{len(inscritos)}", DEF if ok <= len(inscritos) * 0.25 else LIM,
                "Sin detector, la foto entera no se parece al recorte de la cara.")

        # ================================================================ C
        G = "C. Saltarse detector y extractor (_t)"
        malos = {"vector vacío": [], "tamaño incorrecto": np.zeros(100), "NaN": np.full(TAM_TEMPLATE, np.nan),
                 "infinitos": np.full(TAM_TEMPLATE, np.inf), "negativos": -np.ones(TAM_TEMPLATE) / N_CASILLAS,
                 "todo ceros": np.zeros(TAM_TEMPLATE), "todo unos": np.ones(TAM_TEMPLATE),
                 "texto": "hola", "None": None}
        res = [intenta(lambda m=m: s.verificar_t(victima, m))[0] for m in malos.values()]
        npy_pickle = tmp / "pickle.npy"
        np.save(npy_pickle, np.array([{"rce": "os.system"}], dtype=object), allow_pickle=True)
        res.append(intenta(lambda: s.verificar_t(victima, npy_pickle))[0])
        inf.add(G, "Templates malformados (vacío, tamaño, NaN, inf, negativos, ceros, texto, .npy con pickle)",
                f"rechazados {sum(res)}/{len(res)}", DEF if all(res) else LIM,
                "validar_template exige 5900 valores en [0,1] y cada histograma sumando 1; .npy sin pickle.")
        aleat = plantillas_aleatorias(200, rng)
        acc = sum(s.verificar_t(victima, t) for t in aleat)
        acc_i = sum(s.identificar_t(t) is not None for t in aleat)
        inf.add(G, "Templates aleatorios válidos (fuerza bruta, 200 intentos)",
                f"verificar_t aceptó {acc}/200; identificar_t dio nombre {acc_i}/200", DEF if acc + acc_i == 0 else LIM)
        # Template lobo: media de las caras del atacante (no las inscritas)
        publicas = [extraer_template(img) for p in inscritos for img in imgs[p][1:]]
        lobo = np.mean(publicas, axis=0).astype(np.float32)
        acc = sum(s.verificar_t(p, lobo) for p in inscritos)
        d_sin = distancias(lobo, t_inscritos)
        inf.add(G, "Template 'lobo' (media de muchas caras)",
                f"aceptado {acc}/{len(inscritos)} (sin la defensa sería {int((d_sin <= s.umbral).sum())}/{len(inscritos)})",
                DEF if acc == 0 else LIM, "Defensa: distancia mínima al centro de las caras y máximo de coincidencias.")
        lanzo, nom = intenta(lambda: s.inscribir_t("lobo", lobo))
        inf.add(G, "Inscribir un template lobo con inscribir_t para robar identificaciones", nom,
                DEF if lanzo else LIM)
        # Lobo disimulado: mezcla con una cara real
        filas_mezcla = []
        for alfa in (0.3, 0.2, 0.15, 0.1, 0.05):
            acc = tot = 0
            for c in publicas[:10]:
                t = (alfa * lobo + (1 - alfa) * c).astype(np.float32)
                for p in inscritos:
                    tot += 1
                    acc += s.verificar_t(p, t)
            filas_mezcla.append(f"α={alfa}: {100 * acc / tot:.0f} %")
        base = sum(s.verificar_t(p, c) for c in publicas[:10] for p in inscritos) / (10 * len(inscritos))
        inf.add(G, "Lobo disimulado (α·lobo + (1-α)·cara real)",
                "; ".join(filas_mezcla) + f"; cara real sola: {100 * base:.0f} %", LIM,
                "Las mezclas pasan los filtros y aciertan bastante más que una cara real (hasta ~3 veces). "
                "Es el precio de bajar el margen anti-lobo a 0,12 y no aplicar la regla de coincidencias al "
                "verificar (con 0,16 y la regla, ninguna mezcla con α ≥ 0,15 entraba). Ver limitación 7 del README.")
        # Template de la foto pública de la víctima
        ok = sum(s.verificar_t(p, extraer_template(imgs[p][0])) for p in inscritos)
        inf.add(G, "Template calculado de la foto pública de la víctima (el código es abierto)",
                f"aceptado {ok}/{len(inscritos)}", LIM, "Equivale a A2: si la foto es pública, su template también.")
        # Hill-climbing con puntuación
        objetivo = t_inscritos[0]
        pool = np.stack(publicas).reshape(-1, N_CELDAS, N_CASILLAS)
        actual = pool[rng.integers(len(pool))].copy()
        consultas = 0
        def dist(t):
            return float(distancias(t.ravel(), objetivo[None])[0])
        mejor = dist(actual)
        aceptado_en = None
        for _ in range(4):
            for celda in rng.permutation(N_CELDAS):
                for cand in pool[rng.choice(len(pool), 8, replace=False), celda]:
                    prueba = actual.copy()
                    prueba[celda] = cand
                    consultas += 1
                    d = dist(prueba)
                    if d < mejor:
                        actual, mejor = prueba, d
                if aceptado_en is None and s.verificar_t(victima, actual.ravel()):
                    aceptado_en = consultas
            if aceptado_en:
                break
        inf.add(G, "Hill-climbing si el atacante VE la distancia (cambia celdas y se queda con lo que acerca)",
                f"aceptado tras {aceptado_en} consultas" if aceptado_en else f"no aceptado tras {consultas} consultas",
                LIM if aceptado_en else DEF,
                "Por eso los métodos solo devuelven sí/no y la CLI no tiene ninguna opción que muestre la distancia. "
                "Con solo sí/no no hay gradiente que seguir, y el límite de intentos (G) corta la búsqueda.")

        # ================================================================ D
        G = "D. Leer la base de datos"
        contenido = (tmp / "bd.csv").read_bytes()
        t_claro = extraer_template(foto_victima)
        fuga = t_claro.tobytes()[:32] in contenido
        inf.add(G, "Buscar los templates en claro dentro de bd.csv", "encontrados" if fuga else "no encontrados (cifrados)",
                LIM if fuga else DEF, "AES-256-GCM con nonce aleatorio por fila.")
        try:
            import base64
            campo = contenido.decode().splitlines()[1].split(",", 1)[1]
            crudo = np.frombuffer(base64.b64decode(campo)[12:12 + 4 * TAM_TEMPLATE], dtype="<f4")
            como_template = np.isfinite(crudo).all() and (crudo >= 0).all() and (crudo <= 1).all()
        except Exception:  # noqa: BLE001
            como_template = False
        inf.add(G, "Usar directamente el campo cifrado como template",
                "parece un template válido" if como_template else "no es un template válido", LIM if como_template else DEF)
        (tmp / "robada.key").write_bytes(b"\x00" * 32)
        lanzo, nom = intenta(lambda: SistemaBiometrico(ruta_bd=tmp / "bd.csv", ruta_clave=tmp / "robada.key").listar())
        inf.add(G, "Descifrar con una clave adivinada", nom, DEF if lanzo else LIM)
        inf.add(G, "Ver quién está inscrito", f"los {len(inscritos)} nombres están en claro", LIM,
                "Los ids no se cifran (hacen falta para buscar). En esta práctica los nombres ya son públicos.")
        inf.add(G, "Robar bd.csv Y la clave", "con la clave se descifra todo y se pueden usar con verificar_t", LIM,
                "La clave va en otro fichero (secretos/) o en la variable BIOMETRIA_CLAVE; si se roba, no hay protección.")

        # ================================================================ E
        G = "E. Escribir en la base de datos"
        ruta = tmp / "bd.csv"
        copia = ruta.read_bytes()
        lineas = copia.decode().splitlines()
        cab, filas_bd, firma = lineas[0], lineas[1:-1], lineas[-1]   # la firma es la última línea

        def manipula(nuevo: str):
            ruta.write_text(nuevo)
            r = intenta(lambda: s.verificar(victima, foto_victima))
            ruta.write_bytes(copia)
            return r

        def con_firma(filas):
            return "\n".join([cab] + filas + [firma]) + "\n"

        id1, c1 = filas_bd[0].split(",", 1)
        id2, c2 = filas_bd[1].split(",", 1)
        # template del atacante cifrado con OTRA clave
        otra = SistemaBiometrico(ruta_bd=tmp / "otra" / "bd.csv", ruta_clave=tmp / "otra" / "clave.key")
        otra.inscribir(victima, imgs[inscritos[1]][1])
        campo_atacante = (tmp / "otra" / "bd.csv").read_text().splitlines()[1].split(",", 1)[1]
        otro_char = "A" if c1[-3] != "A" else "B"
        casos = {
            "Cambiar un carácter de un template": con_firma([f"{id1},{c1[:-3]}{otro_char}{c1[-2:]}"] + filas_bd[1:]),
            "Intercambiar templates entre dos personas": con_firma([f"{id1},{c2}", f"{id2},{c1}"] + filas_bd[2:]),
            "Renombrar una fila (mi template con el nombre de la víctima)": con_firma([f"{id1},{c1}", f"{id1} ,{c2}"] + filas_bd[2:]),
            "Borrar una fila": con_firma(filas_bd[1:]),
            "Duplicar una fila": con_firma(filas_bd + [filas_bd[0]]),
            "Reordenar las filas": con_firma(filas_bd[::-1]),
            "Añadir una fila con mi template cifrado con otra clave": con_firma(filas_bd + [f"Atacante,{campo_atacante}"]),
            "Sustituir el template de la víctima por el mío (otra clave)": con_firma([f"{id1},{campo_atacante}"] + filas_bd[1:]),
            "Vaciar la base de datos (solo cabecera y firma)": f"{cab}\n{firma}\n",
            "Quitar la línea de firma": "\n".join([cab] + filas_bd) + "\n",
        }
        for nombre, nuevo in casos.items():
            lanzo, nom = manipula(nuevo)
            inf.add(G, nombre, nom, DEF if lanzo else LIM,
                    "HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado).")
        import hashlib
        import hmac as _hmac
        falso = "\n".join([cab, f"{id1},{campo_atacante}"] + filas_bd[1:]) + "\n"
        mac_falso = _hmac.new(b"\x00" * 32, falso.encode(), hashlib.sha256).hexdigest()
        lanzo, nom = manipula(falso + f"#firma,999:{mac_falso}\n")
        inf.add(G, "Modificar y recalcular la firma con una clave inventada", nom, DEF if lanzo else LIM)
        # Rollback: volver a una versión anterior válida (antes de inscribir al último)
        tmp_rb = tmp / "rb"
        tmp_rb.mkdir()
        shutil.copy(tmp / "clave.key", tmp_rb / "clave.key")
        s_rb = SistemaBiometrico(ruta_bd=tmp_rb / "bd.csv", ruta_clave=tmp_rb / "clave.key")
        for p in inscritos[:-1]:
            s_rb.inscribir(p, imgs[p][0])
        ruta.write_bytes((tmp_rb / "bd.csv").read_bytes())
        try:
            n = len(s.listar())
            res, est = f"aceptada: la BD vuelve a tener {n} personas en vez de {len(inscritos)}", LIM
        except BaseDatosComprometidaError as e:
            res, est = type(e).__name__, DEF
        ruta.write_bytes(copia)
        inf.add(G, "Rollback: restaurar una copia anterior válida (con su firma)", res, est,
                "Contador de versión dentro de la firma y copiado en secretos/versiones.json: "
                "una copia antigua tiene versión menor. Si el atacante también controla secretos/, no sirve.")
        ruta.unlink()
        lanzo, nom = intenta(lambda: s.listar())
        ruta.write_bytes(copia)
        inf.add(G, "Borrar la BD entera para empezar de cero", nom, DEF if lanzo else LIM,
                "El contador de secretos/versiones.json sabe que había datos.")
        lanzo, nom = manipula("basura\x00\xff")
        inf.add(G, "Corromper la BD (denegación de servicio)", f"el sistema se detiene ({nom})", LIM,
                "Se prefiere fallar cerrado: mejor no funcionar que funcionar con datos manipulados. "
                "Recuperación: restaurar la copia de seguridad.")
        # Escribir usando la propia API
        lanzo, nom = intenta(lambda: s.inscribir(victima, imgs[inscritos[1]][1]))
        inf.add(G, "Reinscribir a la víctima con mi cara (vía inscribir)", nom, DEF if lanzo else LIM,
                "No se permite reinscribir un id existente (tampoco cambiando mayúsculas).")
        lanzo, nom = intenta(lambda: s.inscribir("Suplantador", foto_victima))
        inf.add(G, "Inscribir la foto de la víctima con otro nombre", nom, DEF if lanzo else LIM,
                f"Regla de duplicados: distancia <= {config.UMBRAL_DUPLICADO} a alguien ya inscrito.")
        otra_foto = imgs[victima][1] if len(imgs[victima]) > 1 else None
        if otra_foto is not None:
            lanzo, nom = intenta(lambda: s.inscribir("Suplantador 2", otra_foto))
            inf.add(G, "Inscribir OTRA foto de la víctima con otro nombre", nom, DEF if lanzo else LIM,
                    "Dos fotos distintas de la misma persona no se distinguen de dos personas parecidas.")

        # ================================================================ F
        G = "F. Entradas maliciosas"
        ids = ["=HYPERLINK(\"http://evil\")", "+1+1", "-2", "@SUM(A1)", "a,b", "a\nb", "a\"b",
               "x" * 65, "", "../../etc/passwd", "_x", 123, None]
        res = [intenta(lambda i=i: s.inscribir_t(i, aleat[0]))[0] for i in ids]
        inf.add(G, "Ids maliciosos (inyección de fórmulas CSV, comas, saltos de línea, comillas, rutas, muy largos)",
                f"rechazados {sum(res)}/{len(res)}", DEF if all(res) else LIM,
                "Solo letras, números, espacios y . _ ' - ; empieza por letra o número; máx. 64.")
        ficheros = {"vacío": b"", "texto": b"no soy una imagen", "JPEG truncado":
                    cv2.imencode(".jpg", foto_victima)[1].tobytes()[:300], "bomba PNG 60000x60000": png_bomba()}
        res = [intenta(lambda b=b: s.verificar(victima, b))[0] for b in ficheros.values()]
        inf.add(G, "Ficheros de imagen maliciosos (vacío, texto, JPEG truncado, bomba de descompresión)",
                f"rechazados sin colgarse {sum(res)}/{len(res)}", DEF if all(res) else LIM,
                "Límite de 40 Mpx y 25 MB antes de decodificar.")
        enorme = tmp / "enorme.txt"
        enorme.write_text("0.1," * 400000)
        lanzo, nom = intenta(lambda: s.verificar_t(victima, enorme))
        inf.add(G, "Fichero de template enorme", nom, DEF if lanzo else LIM, "Límite de 1 MB.")

        # ================================================================ G
        G = "G. Encontrados atacando como el rival"
        import os
        import subprocess
        import sys
        from multiprocessing import get_context

        import biometria
        imp_img = imgs[inscritos[1]][1]
        # G1 umbral desde fuera
        try:
            biometria.main(["--umbral", "5", "listar"])
            res = "aceptada la opción"
        except SystemExit:
            res = "la opción --umbral no existe"
        inf.add(G, "Subir el umbral desde la línea de comandos (--umbral 5)", res,
                DEF if "no existe" in res else LIM, "Antes: aceptaba a cualquiera.")
        entorno = dict(os.environ, BIOMETRIA_UMBRAL="5", BIOMETRIA_BD=str(tmp / "bd.csv"),
                       BIOMETRIA_CLAVE_FICHERO=str(tmp / "clave.key"))
        foto_imp = tmp / "impostor.jpg"
        cv2.imwrite(str(foto_imp), imp_img)
        salida_cli = subprocess.run([sys.executable, str(Path(__file__).with_name("biometria.py")),
                                     "verificar", victima, str(foto_imp)],
                                    env=entorno, capture_output=True, text=True).stdout.strip()
        inf.add(G, "Subir el umbral con la variable de entorno BIOMETRIA_UMBRAL=5 (impostor)", salida_cli,
                DEF if salida_cli == "RECHAZADO" else LIM, "El umbral es una constante; solo se puede bajar desde Python.")
        lanzo, nom = intenta(lambda: setattr(s, "umbral", 5))
        inf.add(G, "Subir el umbral desde Python (s.umbral = 5)", "ValueError" if not lanzo and "ValueError" in nom else nom,
                DEF if "ValueError" in nom else LIM, "Solo se permiten umbrales más estrictos.")
        for opcion in ("--bd", "--detalle"):
            try:
                biometria.main([opcion, "x", "listar"] if opcion == "--bd" else [opcion, "listar"])
                res = "aceptada"
            except SystemExit:
                res = "la opción no existe"
            inf.add(G, {"--bd": "Usar una BD fabricada por el atacante (--bd / --clave)",
                        "--detalle": "Pedir la distancia para hacer hill-climbing (--detalle)"}[opcion], res,
                    DEF if "no existe" in res else LIM)
        # G2 ids homógrafos y casi iguales
        res = []
        for falso in ("А" + victima[1:] if victima[0].upper() == "A" else "А" + victima,
                      victima.upper(), victima.replace("_", "-"), victima.replace("_", "  "),
                      victima.replace("i", "ı"), "Á" + victima[1:]):
            res.append(intenta(lambda f=falso: s.inscribir(f, imp_img))[0])
        inf.add(G, "Inscribirse con un nombre que se ve igual que el de la víctima (A cirílica, ı turca, "
                   "mayúsculas, guiones, dobles espacios, tildes)",
                f"rechazados {sum(res)}/{len(res)}", DEF if all(res) else LIM,
                "Solo alfabeto latino, y los ids se comparan sin mayúsculas, tildes ni separadores. "
                "No cubre I mayúscula, 1 o 0 en lugar de l u o (ver limitación 19 del README).")
        # G3 diccionario: templates de fotos públicas de OTRAS personas
        def diccionario(sistema):
            exitos, intentos_ok = 0, []
            for v in inscritos:
                candidatos = [extraer_template(img) for p in inscritos if p != v for img in imgs[p]]
                for k, t in enumerate(candidatos, 1):
                    if sistema.verificar_t(v, t):
                        exitos += 1
                        intentos_ok.append(k)
                        break
            return exitos, intentos_ok
        e1, k1 = diccionario(s)
        inf.add(G, "Diccionario SIN límite: probar los templates de las fotos públicas de otros famosos",
                f"entra como {e1}/{len(inscritos)} víctimas (mediana {np.median(k1) if k1 else '-'} intentos)", LIM,
                "Consecuencia directa de la FAR del 19 %: con intentos ilimitados casi siempre se entra.")
        s_lim = SistemaBiometrico(ruta_bd=tmp / "bd.csv", ruta_clave=tmp / "clave.key", limitar_intentos=True)
        e2, k2 = diccionario(s_lim)
        inf.add(G, f"Diccionario CON el límite de intentos ({config.MAX_FALLOS_SEGUIDOS} fallos seguidos → bloqueo)",
                f"entra como {e2}/{len(inscritos)} víctimas", LIM if e2 else DEF,
                f"El bloqueo dura {config.BLOQUEO_SEGUNDOS} s y se duplica cada vez; corta la búsqueda, "
                f"pero quien acierta en los primeros intentos entra (FAR por intento ~19 %). "
                f"El límite solo cuenta en verificar (no en identificar) y no garantiza el máximo con "
                f"intentos simultáneos: ver limitaciones 9 y 20 del README.")
        for t in aleat[:config.MAX_FALLOS_SEGUIDOS]:      # el atacante falla a propósito
            s_lim.verificar_t(victima, t)
        ok = s_lim.verificar(victima, foto_victima)
        inf.add(G, "Bloquear a la víctima a propósito (denegación de servicio)",
                "la víctima legítima queda rechazada" if not ok else "la víctima sigue entrando", LIM if not ok else DEF,
                "Precio del límite de intentos: el atacante puede bloquear una cuenta unos minutos.")
        # G4 morphing
        from preprocesado import _matriz_semejanza

        def alinear(img):
            _, oi, od = detectar(img)
            return cv2.warpAffine(img, _matriz_semejanza(oi, od, 256), (256, 256))
        ambos = tot = 0
        for i_ in range(0, len(inscritos) - 1, 2):
            a_, b_ = inscritos[i_], inscritos[i_ + 1]
            m = cv2.addWeighted(alinear(imgs[a_][0]), 0.5, alinear(imgs[b_][0]), 0.5, 0)
            m = cv2.copyMakeBorder(m, 64, 64, 64, 64, cv2.BORDER_REPLICATE)
            try:
                tot += 1
                ambos += s.verificar(a_, m) and s.verificar(b_, m)
            except ErrorDetector:
                pass
        inf.add(G, "Morphing: mezcla 50/50 de dos caras", f"aceptada como LAS DOS personas en {ambos}/{tot}", LIM,
                "Ataque conocido en biometría facial; detectar morphs requiere modelos específicos.")
        # G5 concurrencia
        dir_c = tmp / "concurrencia"
        dir_c.mkdir()
        with get_context("spawn").Pool(4) as pool:
            pool.map(_inscribir_en_paralelo, [(dir_c, i) for i in range(4)])
        try:
            n = len(SistemaBiometrico(ruta_bd=dir_c / "bd.csv", ruta_clave=dir_c / "clave.key").listar())
            res, est = f"{n}/24 inscritos, BD íntegra", DEF if n == 24 else LIM
        except BaseDatosComprometidaError as e:
            res, est = f"BD corrompida ({e})", LIM
        inf.add(G, "4 procesos inscribiendo a la vez (antes corrompía la BD para siempre)", res, est,
                "Bloqueo de fichero entre procesos (bloqueo.py) en cada lectura y escritura.")
        # G6 inscripción masiva
        lanzo = False
        with config_temporal(MAX_INSCRITOS=len(inscritos) + 3):
            for k_, t in enumerate(plantillas_aleatorias(10, rng)):
                ok_, nom = intenta(lambda t=t, k_=k_: s.inscribir_t(f"spam {k_}", t))
                if ok_:
                    lanzo = True
                    break
        inf.add(G, "Inscripción masiva para ralentizar el sistema", f"cortada al llegar al máximo ({nom})" if lanzo else "sin límite",
                DEF if lanzo else LIM, f"Máximo de {config.MAX_INSCRITOS} inscritos (con 300 filas cada operación tardaba ~0,5 s).")
        # G7 fuga de ficheros
        secreto = tmp / "secreto.txt"
        secreto.write_text("contraseña_super_secreta")
        try:
            s.verificar_t(victima, secreto)
            res = "sin error"
        except ErrorTemplate as e:
            res = "el error muestra el contenido" if "contraseña" in str(e) else "error sin contenido del fichero"
        lanzo, nom = intenta(lambda: s.verificar_t(victima, "/etc/passwd"))
        inf.add(G, "Leer ficheros ajenos con verificar_t (antes el error mostraba su primera línea)",
                f"{res}; /etc/passwd → {nom}", DEF if "sin contenido" in res and lanzo else LIM,
                "Solo .npy/.csv/.txt y mensajes de error genéricos.")

        # ================================================================ H
        G = "H. Encontrados por la revisión externa"
        import struct as _st
        bomba = tmp / "bomba25k.png"

        def _trozo(t, d):
            return _st.pack(">I", len(d)) + t + d + _st.pack(">I", zlib.crc32(t + d))
        comp = zlib.compressobj(9)
        fila_png = b"\x00" + b"\x80" * 25000
        bomba.write_bytes(b"\x89PNG\r\n\x1a\n" + _trozo(b"IHDR", _st.pack(">IIBBBBB", 25000, 25000, 8, 0, 0, 0, 0))
                          + _trozo(b"IDAT", b"".join(comp.compress(fila_png) for _ in range(25000)) + comp.flush())
                          + _trozo(b"IEND", b""))
        codigo = ("import cv2, sistema\n"
                  f"try: sistema.extraer_template({str(bomba)!r})\n"
                  "except sistema.ImagenInvalidaError: print('RECHAZADA')\n"
                  "print(int([l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]) // 1024)\n")
        # Memoria medida con VmHWM (empieza de cero en el proceso nuevo; ru_maxrss
        # heredaría el pico de este proceso). Solo en Linux.
        out = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True,
                             cwd=str(Path(__file__).parent)).stdout.split() if Path("/proc/self/status").exists() else []
        ok = len(out) == 2 and out[0] == "RECHAZADA" and int(out[1]) < 600
        inf.add(G, f"PNG de {bomba.stat().st_size // 1024} KB que dice medir 25000x25000 (importando cv2 antes)",
                f"rechazado con un pico de {out[-1] if out else '?'} MB (antes 3,7 GB)", DEF if ok else LIM,
                "Se leen las dimensiones de la cabecera (Pillow) antes de decodificar.")
        # Corte a mitad de escritura
        import basedatos as _bd
        dir_corte = tmp / "corte"
        s_c = SistemaBiometrico(ruta_bd=dir_corte / "bd.csv", ruta_clave=dir_corte / "clave.key")
        s_c.inscribir_t("A", aleat[0])
        original = _bd._escritura_atomica
        sobrevive = True
        for corte_en in (1, 2):
            llamadas = [0]

            def con_corte(r, c, corte_en=corte_en, llamadas=llamadas):
                llamadas[0] += 1
                if llamadas[0] == corte_en:
                    raise KeyboardInterrupt
                original(r, c)
            _bd._escritura_atomica = con_corte
            try:
                s_c.inscribir_t(f"B{corte_en}", aleat[corte_en])
            except KeyboardInterrupt:
                pass
            finally:
                _bd._escritura_atomica = original
            try:
                SistemaBiometrico(ruta_bd=dir_corte / "bd.csv", ruta_clave=dir_corte / "clave.key").listar()
            except BaseDatosComprometidaError:
                sobrevive = False
        inf.add(G, "Ctrl+C / apagón a mitad de una inscripción (antes dejaba la BD inservible para siempre)",
                "la BD sigue funcionando" if sobrevive else "BD inutilizada", DEF if sobrevive else LIM,
                "La firma va dentro del .csv: un único reemplazo atómico por escritura.")
        # CRLF de git
        dir_crlf = tmp / "crlf"
        s_c = SistemaBiometrico(ruta_bd=dir_crlf / "bd.csv", ruta_clave=dir_crlf / "clave.key")
        s_c.inscribir_t("A", aleat[0])
        r_ = dir_crlf / "bd.csv"
        r_.write_bytes(r_.read_bytes().replace(b"\n", b"\r\n"))
        lanzo, nom = intenta(lambda: s_c.listar())
        inf.add(G, "Subir la BD con git en Windows (core.autocrlf convierte \\n en \\r\\n)",
                "BD rechazada" if lanzo else "BD aceptada", LIM if lanzo else DEF,
                "Se normalizan los saltos de línea antes de comprobar la firma; clave en hexadecimal; .gitattributes.")
        # Dos BD con la misma clave
        otra_bd = SistemaBiometrico(ruta_bd=tmp / "otra2" / "bd2.csv", ruta_clave=tmp / "clave.key",
                                    limitar_intentos=False)
        lanzo, nom = intenta(lambda: (otra_bd.listar(), otra_bd.inscribir_t("X", aleat[5]), s.listar()))
        inf.add(G, "Usar una segunda BD con la misma clave (antes: 'falta bd.csv... ¿la han borrado?')",
                "error" if lanzo else "funcionan las dos", LIM if lanzo else DEF, "Un contador de versión por BD.")
        # Entradas raras
        npy = tmp / "cabecera_falsa.npy"
        np.save(npy, np.zeros(TAM_TEMPLATE, np.float32))
        npy.write_bytes(npy.read_bytes().replace(b"(5900,), }", b"(999999999999999999999,), }"))
        res = [intenta(lambda: s.verificar_t(victima, npy))[0],
               intenta(lambda: s.verificar(victima, np.empty((100, 100, 3), dtype=object)))[0]]
        inf.add(G, "Excepciones sin controlar: .npy con cabecera de tamaño absurdo, foto con dtype=object",
                f"controladas {sum(res)}/{len(res)}", DEF if all(res) else LIM)
        # Persona NO inscrita que intenta identificarse
        con_nombre = total = 0
        for fuera in inscritos:
            d_ab = tmp / f"abierto_{inscritos.index(fuera)}"
            s_ab = SistemaBiometrico(ruta_bd=d_ab / "bd.csv", ruta_clave=d_ab / "clave.key", limitar_intentos=False)
            for p in inscritos:
                if p != fuera:
                    s_ab.inscribir_t(p, t_inscritos[inscritos.index(p)])
            for img in imgs[fuera]:
                total += 1
                con_nombre += s_ab.identificar(img) is not None
        inf.add(G, "Persona NO inscrita usa identificar (por ejemplo, alguien del grupo rival)",
                f"recibe el nombre de otro en {con_nombre}/{total} ({100 * con_nombre / total:.0f} %)", LIM,
                f"Es la FAR en modo identificación, con {len(inscritos) - 1} inscritos; con 51 será mayor. "
                "Limitación de LBP; se reduce con un umbral más estricto.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    cab = (f"# Evaluación de seguridad (ataques)\n\nSistema: LBP, umbral {config.UMBRAL}. "
           f"BD temporal con {len(inscritos)} personas (1ª foto de cada una); el resto de fotos se usan "
           f"como material del atacante.")
    texto = inf.markdown(cab)
    (salida / "informe_ataques.md").write_text(texto, encoding="utf-8")
    print(f"\nInforme en {salida / 'informe_ataques.md'}")


if __name__ == "__main__":
    main()
