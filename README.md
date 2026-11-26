# Sistema de reconocimiento facial con LBP

Práctica de Sistemas Biométricos. Aproximación asignada: **texturas, LBP (Local Binary Patterns)**.

```
foto ─► detector (YuNet) ─► preprocesado ─► extractor LBP ─► template ─► comparación ─► decisión
        caja + ojos         alinear + luz    5900 valores     (cifrado    chi-cuadrado    umbral
                                                              en .csv)
```

- Python, base de datos **.csv**, Docker listo para usar.
- Los **9 métodos** del enunciado: `inscribir`, `verificar`, `identificar` y sus versiones `_sd` (sin detector) y `_t` (con template).
- Base de datos **cifrada (AES-256-GCM) y firmada (HMAC-SHA256)**, con protección contra *rollback* y contra escrituras simultáneas.
- Límite de intentos al verificar, nombres sin homógrafos de otros alfabetos, umbral no manipulable desde fuera. Lo que estas defensas **no** cubren está en la sección 6.
- Evaluación de rendimiento (`evaluar.py`) y de seguridad con **61 ataques** (`ataques.py`): 13 encontrados atacando el sistema como lo haría el grupo rival y 7 a partir de una revisión externa. Los informes están en `resultados/`.

---

## 1. Puesta en marcha

### Con Docker (recomendado)

```bash
docker build -t biometria-lbp .
```

Crea, junto al proyecto, una carpeta `fotos/` con las fotos. **En Linux y macOS, crea también `datos/` y `secretos/` antes del primer uso**: si no existen, Docker las crea como `root` y el contenedor no puede escribir en ellas, aunque uses `--user`. En Windows no hace falta.

```bash
# Linux / macOS (solo la primera vez)
mkdir -p datos secretos fotos
```

Después:

```bash
# Linux / macOS
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/datos:/app/datos" -v "$PWD/secretos:/app/secretos" -v "$PWD/fotos:/fotos:ro" \
  biometria-lbp verificar "Angelina Jolie" /fotos/prueba.jpg

# Windows (PowerShell)
docker run --rm -v "${PWD}/datos:/app/datos" -v "${PWD}/secretos:/app/secretos" -v "${PWD}/fotos:/fotos:ro" `
  biometria-lbp verificar "Angelina Jolie" /fotos/prueba.jpg
```

O, más corto, con Docker Compose (monta `datos/`, `secretos/` y `fotos/` solo; en Linux y macOS, con las carpetas ya creadas como arriba):

```bash
docker compose run --rm biometria verificar "Angelina Jolie" /fotos/prueba.jpg

# Linux, si tu usuario no es el 1000 (compruébalo con: id -u)
docker compose run --rm --user "$(id -u):$(id -g)" biometria verificar "Angelina Jolie" /fotos/prueba.jpg
```

### Sin Docker

```bash
pip install -r requirements.txt
python -m pytest -q                 # 104 pruebas + 1 limitación conocida marcada
python biometria.py --help
```

Probado con Python 3.13 y OpenCV 4.10, 5.0 y 5.1. Las rutas con `º`, tildes o `ñ` (Windows) están soportadas.

---

## 2. Inscribir a las 51 personas

Pon las 50 fotos del profesor y la vuestra en una carpeta, en cualquiera de estos dos formatos:

```
inscripcion/                        inscripcion/
  Angelina Jolie.jpg                  Angelina Jolie/foto.jpg
  Brad Pitt.jpg            o bien     Brad Pitt/foto.jpg
  ...                                 ...
```

El **id** es el nombre del fichero (o de la subcarpeta).

```bash
python biometria.py inscribir_carpeta inscripcion/
python biometria.py listar          # debe decir Total: 51
```

Si alguna foto da `SinCaraError`, recórtale la cara a mano e inscríbela con `inscribir_sd`.

La primera inscripción crea `secretos/clave.key` (la clave), `secretos/versiones.json` (anti-rollback) y, al verificar, `secretos/intentos.json` (límite de intentos). **Sin esa carpeta la base de datos no se puede leer:** guardadla y entregadla junto con `datos/`.

---

## 3. Los 9 métodos

### Desde Python

```python
from sistema import *

inscribir("Angelina Jolie", "fotos/angelina.jpg")    # True (o excepción si no se puede)
verificar("Angelina Jolie", "fotos/prueba.jpg")      # True / False
identificar("fotos/prueba.jpg")                      # "Angelina Jolie" / None

inscribir_sd(id, cara_recortada)                     # sin detector: la foto ya es la cara
verificar_sd(id, cara_recortada)
identificar_sd(cara_recortada)

t = extraer_template("fotos/prueba.jpg")             # numpy float32, 5900 valores
inscribir_t(id, t)                                   # sin detector ni extractor
verificar_t(id, t)                                   # también acepta ruta .npy / .csv / .txt
identificar_t(t)
```

`foto` puede ser una ruta, los bytes del fichero o un array numpy BGR. Para usar otra base de datos u otro umbral: `s = SistemaBiometrico(ruta_bd=..., umbral=...)`; `s.ultimo` guarda el detalle de la última decisión (distancia y motivo).

### Desde la línea de comandos

```bash
python biometria.py inscribir      "Angelina Jolie" foto.jpg
python biometria.py verificar      "Angelina Jolie" foto.jpg
python biometria.py identificar    foto.jpg
python biometria.py inscribir_sd   "Angelina Jolie" cara.jpg
python biometria.py verificar_sd   "Angelina Jolie" cara.jpg
python biometria.py identificar_sd cara.jpg
python biometria.py extraer        foto.jpg template.npy      # [--sd]
python biometria.py inscribir_t    "Angelina Jolie" template.npy
python biometria.py verificar_t    "Angelina Jolie" template.npy
python biometria.py identificar_t  template.npy
python biometria.py listar
```

Códigos de salida: `0` aceptado / hecho, `1` rechazado / no identificado, `2` error.

A propósito, la línea de comandos **no** tiene opciones para cambiar el umbral, la base de datos o la clave, ni para ver la distancia (ver la sección 6). La distancia solo se puede ver desde Python (`s.ultimo`), para los scripts de evaluación.

### Reglas de decisión

| Situación | Respuesta |
|---|---|
| Distancia al template del id ≤ umbral (0,345) | `verificar` → `True` |
| Id no inscrito | `False` (igual que un rechazo: no revela qué ids existen) |
| 3 fallos seguidos contra el mismo id | el id queda bloqueado 5 min (10, 20, 40… si se repite) y responde `False` aunque la cara sea buena; un acierto reinicia la cuenta |
| `identificar`: el más cercano está por debajo del umbral **y** gana al segundo por ≥ 0,01 | su id |
| Si no | `None` ("no identificado"; mejor no dar nombre que dar uno falso) |
| Foto sin cara / fichero inválido / template malformado | excepción con un mensaje claro |
| Template sospechoso de ser "lobo" | rechazo, y aviso en el registro |
| Inscribir un id que ya existe, aunque cambie en mayúsculas, tildes o separadores (`angelina_jolie` = `Angelina Jolie` = `ÁNGELINA-JOLIE`) | `YaInscritoError` (no se puede "secuestrar" una identidad) |
| Id con letras no latinas (por ejemplo, una A cirílica idéntica a la vista) | `IdInvalidoError` |
| Inscribir una cara casi idéntica a otra ya inscrita | `InscripcionRechazadaError` |

---

## 4. Cómo funciona

### 4.1 Detector: `detector.py`
**YuNet** (`cv2.FaceDetectorYN`), una CNN ligera de OpenCV (230 KB, va en CPU) que da la caja de la cara y 5 puntos: ojos, nariz y comisuras de la boca. Con varias caras se queda con la más grande. Valida la entrada antes de decodificarla (máx. 25 MB y 40 Mpx, contra bombas de descompresión), respeta la orientación EXIF de las fotos de móvil y carga el modelo en memoria, así que funciona en rutas con caracteres especiales en Windows.

### 4.2 Preprocesado: `preprocesado.py`
1. **Alineación** por los ojos: una transformación de semejanza los lleva a posiciones fijas, (0,27; 0,32) y (0,73; 0,32) en una cara de 128×128. Así cada celda de la rejilla cae siempre sobre la misma zona de la cara. El recorte es ajustado para dejar fuera el pelo y el fondo.
2. **Escala de grises.**
3. **Normalización de iluminación de Tan y Triggs** (corrección gamma, diferencia de gaussianas y ecualización de contraste).

En la vía `_sd` no hay detector que dé los ojos, así que se colocan en su **posición media dentro de la caja de YuNet** (medida sobre 61 fotos reales). Así, una cara recortada se alinea igual que en la vía normal.

### 4.3 Extractor LBP: `lbp.py`
1. Cada píxel se compara con **8 vecinos en un círculo de radio 2**: vecino ≥ centro → 1. Los 8 bits forman un código de textura local (borde, esquina, mancha...).
2. Solo se distinguen los patrones **uniformes** (como mucho 2 cambios 0↔1); el resto va a una casilla común. Salen **59 casillas**.
3. La cara se divide en una **rejilla de 10×10** y se hace un histograma por celda, para conservar *dónde* está cada textura.
4. **Template simétrico:** se promedian los histogramas de la cara y de su imagen en espejo.
5. Template = 100 histogramas × 59 casillas = **5900 valores**.
6. **Comparación** con la distancia chi-cuadrado media por celda (0 = idénticos, 2 = nada en común).

### 4.4 Por qué estos parámetros (experimentos con 61 fotos reales de 13 famosos)

| Cambio | EER |
|---|---|
| Versión inicial (CLAHE, ojos a 0,30/0,36, rejilla 8×8) | 30,5 % |
| + normalización Tan-Triggs | ~21 % |
| + recorte más ajustado (ojos a 0,27/0,32) | ~20 % |
| + rejilla 10×10 + template simétrico | **19,5 %** |
| Probado y descartado: radios 1 y 3, LBP multiescala, 16 vecinos, caras de 144 y 160 px, distancia de Hellinger, pesos por región (ojos ×4, boca ×2) | sin mejora clara |

### 4.5 Base de datos: `basedatos.py`

```
datos/bd.csv              id,template                <- template = base64(nonce + AES-256-GCM(5900 float32))
                          ...una fila por persona...
                          #firma,<versión>:<HMAC-SHA256 de todo lo anterior>
secretos/clave.key        clave maestra (32 bytes en hexadecimal) <- fuera de datos/ (o variable BIOMETRIA_CLAVE)
secretos/versiones.json   contador anti-rollback, uno por BD
secretos/intentos.json    fallos seguidos y bloqueos por id
```

- **Confidencialidad:** quien lea `bd.csv` solo ve los nombres, no los templates.
- **Integridad por fila:** GCM autentica cada template junto con su id. Cambiar un byte o mover un template a otro nombre se detecta.
- **Integridad del fichero:** la última línea es un HMAC de todo el .csv, así que borrar, duplicar, reordenar o añadir filas se detecta.
- **Anti-rollback:** la versión va dentro del HMAC y además se guarda en `secretos/versiones.json`, así que restaurar una copia antigua (aunque sea válida) se detecta. Hay un contador por BD, así que varias BD pueden compartir la clave.
- **Falla cerrado:** ante cualquier manipulación lanza `BaseDatosComprometidaError` y no funciona.
- **Escritura atómica de un solo fichero:** se escribe en un temporal y luego un único `os.replace`. Un corte (Ctrl+C, apagón) deja la versión anterior o la nueva, nunca la BD rota. (Con la firma en un fichero aparte, un corte entre las dos escrituras la dejaba inservible.)
- **Tolerante a git en Windows:** los saltos de línea se normalizan antes de comprobar la firma y la clave es texto hexadecimal; `.gitattributes` evita conversiones de todos modos.
- **Bloqueo entre procesos** (`bloqueo.py`) en cada lectura y escritura. Sin él, dos inscripciones a la vez dejaban el `.csv` y la firma desemparejados y la BD inservible para siempre.
- Las dos subclaves (cifrado y HMAC) se derivan de la maestra con HKDF.

### 4.6 Defensa anti-lobo
Un template "lobo" es la media de muchas caras y se parece a todo el mundo: **sin defensa, se acepta como el 100 % de los inscritos** vía `verificar_t`. El sistema:
- rechaza (al inscribir, verificar e identificar) cualquier template a ≤ 0,12 del **centro de las caras** (la media de la BD y la media de referencia de `models/centro_referencia.npy`). Los lobos están a ~0,02; las caras reales medidas, a ≥ 0,181. Se deja margen porque el centro y estas cifras salen de las mismas 61 fotos;
- además, **solo al inscribir**, rechaza un template que coincida con **más del 75 % de los inscritos** (con 20 o más). Una cara real coincide de media con el 24 %, pero con galerías pequeñas puede pasar del 50 %. Por eso no se aplica al verificar: rechazaría siempre a esa persona legítima.

**Precio de estos dos cambios** (antes: 0,16 y la regla del 50 % también al verificar): ya no se rechaza a gente legítima, pero el lobo disimulado entra mucho más. Con α = 0,15, antes entraba el 0 % y ahora el 76 %, frente al 22 % de una cara real (fila 15 del informe de ataques y limitación 7).

---

## 5. Evaluación

```bash
python evaluar.py fotos_evaluacion/ --salida resultados/rendimiento/
python ataques.py fotos_evaluacion/ --salida resultados/ataques/
python probar_carpeta.py fotos/ --salida res_detector/     # revisar a ojo el detector
```

`fotos_evaluacion/` tiene una subcarpeta por persona con 2 o más fotos (la 1ª se inscribe y el resto son intentos). Con una sola foto por persona (como las del profesor) no hay pares genuinos: añadid fotos vuestras o de otras fuentes.

### Rendimiento: `resultados/rendimiento/informe.md`
Con 61 fotos reales de 13 famosos (fotos no controladas: luz, pose, maquillaje, edad):

| Medida | Valor |
|---|---|
| EER | **19,5 %** (umbral 0,346) |
| Con el umbral del sistema (0,345) | FAR 19,0 %, FRR 21,3 % |
| Umbral para FAR ≤ 1 % | 0,307, con FRR 60 % |
| Protocolo de la práctica (1 foto inscrita): verificar | FRR 31 %, FAR 18 % |
| Protocolo de la práctica: identificar | 35 % correcto, 17 % otra persona, 48 % no identificado |
| **Persona NO inscrita** que intenta identificarse | **recibe el nombre de otra en el 52 %** de los casos (12 inscritos; con 51 será mayor) |
| Fallos del detector | 0 de 61 |

**Aviso de metodología:** las mismas 61 fotos se han usado para elegir los parámetros, calibrar los umbrales y evaluar. No hay un conjunto de prueba aparte, así que las cifras pueden ser algo optimistas. Volved a pasar `evaluar.py` con fotos que el sistema no haya visto.

Para un sistema más estricto, bajad `UMBRAL` en `config.py` (por ejemplo, 0,32): la FAR baja a ~5 % y la FRR sube a ~45 %. Desde fuera no se puede cambiar.

### Seguridad: `resultados/ataques/informe_ataques.md`
**61 ataques: 42 defendidos y 19 limitaciones conocidas.** Resumen:

| Grupo | Defendido | Limitación conocida |
|---|---|---|
| A. Presentación | foto sin cara | FAR 18 %; la foto de inscripción (pública), degradada o en espejo se acepta; varias caras |
| B. Sin detector (`_sd`) | no-caras (0/494), imagen lisa, foto sin recortar | la cara de inscripción recortada se acepta |
| C. Con template (`_t`) | malformados, pickle, aleatorios (0/200), lobo, inscribir lobo | lobo disimulado (hasta 76 % contra 22 % de una cara real), template de foto pública, hill-climbing si se ve la distancia (128 consultas) |
| D. Leer la BD | templates cifrados, campo cifrado inútil, clave adivinada | los nombres se ven; si roban también la clave, no hay protección |
| E. Escribir en la BD | 12 manipulaciones distintas, incluido el rollback y borrar todo; reinscribir; duplicar foto con otro nombre | denegación de servicio (falla cerrado); otra foto de la víctima con otro nombre |
| F. Entradas maliciosas | ids (inyección CSV...), ficheros (bomba PNG...), template enorme | — |
| G. Atacando como el rival | umbral por CLI, por variable de entorno y por Python; BD del atacante; pedir la distancia; nombres homógrafos; 4 procesos a la vez; inscripción masiva; leer ficheros ajenos | diccionario de fotos públicas (12/13 víctimas sin límite, 7/13 con límite, y el límite solo cuenta en `verificar`); bloquear a la víctima; morphing (5/6); nombres con I/1/0 en lugar de l/o |
| H. Revisión externa | bomba PNG de 25000×25000 (antes 3,7 GB de memoria); corte a mitad de escritura; CRLF de git; dos BD con la misma clave; excepciones sin controlar | persona no inscrita identificada como otra (52 %) |

---

## 6. Limitaciones conocidas

1. **Modelo de amenaza:** quien pueda modificar el código o leer `secretos/` controla el sistema por completo. La interfaz que se defiende son los 9 métodos y la línea de comandos; las defensas de la BD protegen frente a quien solo accede a `datos/`.
2. **Precisión de LBP:** con caras no controladas, el EER es ~19,5 %. Es una limitación de la aproximación (textura local sin aprendizaje), no del código: LBP es muy sensible a la pose, la expresión y el maquillaje.
3. **Sin detección de vida:** una foto (o la foto pública de inscripción) se acepta como la persona. Las 50 fotos son comunes a todos los grupos.
4. **Template simétrico:** la foto en espejo da el mismo resultado (es a propósito).
5. **Inclinación de la cabeza:** los ojos de YuNet infravaloran la inclinación a partir de ~10°, así que la alineación queda incompleta en caras muy giradas (más falsos rechazos).
6. **Varias caras:** se usa la más grande.
7. **Lobo disimulado:** una mezcla de lobo y cara real pasa los filtros y acierta bastante más que una cara normal: con α = 0,15 entra el 76 % de las veces, frente al 22 % de una cara real (fila 15 del informe). Es consecuencia de bajar el margen anti-lobo a 0,12 y de no aplicar la regla de coincidencias al verificar, cambios hechos para no rechazar a personas legítimas. Con el margen anterior (0,16 y la regla al verificar), ninguna mezcla con α ≥ 0,15 entraba, pero se rechazaba como "lobo" a algunas caras reales.
8. **Hill-climbing:** si el atacante ve la distancia, en ~130 consultas consigue un template aceptado. Por eso los métodos solo devuelven sí/no y la CLI no muestra distancias; `s.ultimo` existe en Python solo para evaluar.
9. **Ataque de diccionario:** con la FAR del 19 %, probar templates de fotos públicas de otros famosos entra como 12 de 13 víctimas si no hay límite. El límite de 3 fallos lo baja a 7/13 (quien acierta en los primeros intentos entra). **El límite solo se aplica a `verificar`, `verificar_sd` y `verificar_t`:** `identificar`, `identificar_sd` e `identificar_t` no tienen límite de intentos, así que las mismas pruebas pueden hacerse por la vía de identificación sin que cuenten como fallos. Con el umbral 0,32 la probabilidad de éxito por intento bajaría del ~19 % al ~5 %, a cambio de rechazar a casi la mitad de los usuarios legítimos (FRR ~45 %).
10. **Bloqueo como denegación de servicio:** 3 fallos a propósito bloquean a la víctima unos minutos.
11. **Morphing:** una mezcla 50/50 de dos caras se acepta como las dos personas.
12. **Nombres en claro** en `bd.csv`.
13. **Clave:** quien tenga `secretos/` tiene todo (puede descifrar, firmar y restaurar copias antiguas).
14. **Denegación de servicio:** si se corrompe la BD, el sistema se detiene (falla cerrado). Hay que restaurar la copia de seguridad.
15. **Inscripción:** dos fotos distintas de la misma persona con nombres distintos no se distinguen de dos personas parecidas.
16. **Persona no inscrita:** `identificar` le da el nombre de otra persona en el 52 % de los casos (con 12 inscritos; con 51, más). Es la FAR en modo identificación. El margen de identificación ayuda, pero no lo evita.
17. **Calibración con los mismos datos:** parámetros, umbrales y centro anti-lobo salen de las 61 fotos con las que también se evalúa.
18. **Docker en Linux:** las carpetas montadas tienen que existir antes del primer uso (`mkdir -p datos secretos fotos`) y el contenedor tiene que ejecutarse con vuestro usuario (`--user "$(id -u):$(id -g)"`, también con `docker compose run`). Si Docker crea las carpetas, quedan de `root` y el contenedor no puede escribir aunque se use `--user`.
19. **Nombres parecidos dentro del alfabeto latino:** se rechazan otros alfabetos (A cirílica…), las variantes de mayúsculas, tildes y separadores, y algunas letras latinas confundibles (ı, ł…). Pero **no** `I` mayúscula, `1` o `0` en lugar de `l` u `o`: `Angelina JoIie`, `Angelina Jo1ie` y `Angelina J0lie` se pueden inscribir junto a `Angelina Jolie`, y en muchas fuentes (Arial, Calibri) `JoIie` se ve igual que `Jolie`.
20. **Límite de intentos con intentos simultáneos:** la comprobación de si un id está bloqueado y el registro del fallo son dos pasos separados. Varios intentos lanzados a la vez pueden pasar la comprobación antes de que se registre el tercer fallo, así que el máximo de 3 fallos seguidos no está garantizado en ese caso.
21. **`datos/` en solo lectura:** cada lectura de la BD crea un fichero de bloqueo (`datos/.bd.csv.lock`). Si `datos/` está montada o protegida en solo lectura, ninguna operación funciona, tampoco `verificar` (`ERROR inesperado (OSError)`).
22. **Contador anti-rollback por nombre de carpeta:** cada BD se identifica en `secretos/versiones.json` por `carpeta/fichero` (por ejemplo `datos/bd.csv`), para que valga igual en el ordenador y en Docker. Dos copias del proyecto que usen la **misma** carpeta `secretos/` comparten contador, y la segunda falla con "Falta bd.csv pero el contador dice que había datos". Cada copia debe tener su propia carpeta `secretos/`.
23. **Formato de la BD:** la firma va en la última línea del `.csv` (`#firma,<versión>:<HMAC>`), así que un lector de CSV la ve como una fila más, y la clave se guarda en hexadecimal. Una BD creada con una versión anterior del sistema no se puede leer: hay que volver a inscribir a todos.

---

## 7. Qué se entrega al grupo rival

| Qué | Para qué |
|---|---|
| El repositorio (código, `Dockerfile`, README, `resultados/`) | Para ejecutar y estudiar el sistema |
| `datos/bd.csv` (con su firma en la última línea) con las 51 personas inscritas | Es la base de datos que van a atacar |
| `secretos/` (`clave.key` + `versiones.json`) | Sin ella el sistema no arranca, así que hay que dársela |
| La lista de los 51 ids (`python biometria.py listar`) | Para saber a quién intentar suplantar |

Al darles `secretos/` tienen técnicamente todo. Por eso, en la entrega hay que dejar claro el **modelo de amenaza**: los escenarios del enunciado ("lee la BD", "escribe en la BD") se refieren a un atacante que solo accede a `datos/`, y la interfaz que se ataca son los 9 métodos. Que "con la clave se descifra todo" ya está documentado (ataque D22): si lo dicen, no es un fallo desconocido.

## 8. Estructura del repositorio

| Fichero | Contenido |
|---|---|
| `sistema.py` | **Los 9 métodos**, reglas de decisión y defensas |
| `biometria.py` | Línea de comandos (punto de entrada del Docker) |
| `detector.py` | Detector YuNet y validación de imágenes |
| `preprocesado.py` | Alineación + Tan-Triggs; versión sin detector |
| `lbp.py` | Extractor LBP, validación de templates y distancia chi-cuadrado |
| `pipeline.py` | Foto → template (con y sin detector) |
| `basedatos.py` | .csv cifrado y firmado, anti-rollback |
| `bloqueo.py` | Bloqueo de ficheros entre procesos (Linux, macOS, Windows) |
| `config.py` | Umbrales, límites y rutas (las rutas se pueden cambiar con `BIOMETRIA_BD`, `BIOMETRIA_CLAVE`, `BIOMETRIA_CLAVE_FICHERO`; el umbral no) |
| `evaluar.py` / `ataques.py` | Evaluación de rendimiento y de seguridad |
| `probar_carpeta.py` | Pasa el detector por una carpeta y separa ok / sin cara / varias / error |
| `test_*.py` | 105 pruebas automáticas |
| `models/` | Modelo YuNet y centro de referencia anti-lobo |
| `resultados/` | Informes y gráficas generados |
| `Dockerfile`, `docker-compose.yml` | Contenedor |

## 9. Reparto para la exposición

| Persona | Parte | Ficheros |
|---|---|---|
| 1 | Detector y validación de entradas (ataques A, F) | `detector.py`, `probar_carpeta.py` |
| 2 | Preprocesado + LBP + elección de parámetros (sección 4.4) | `preprocesado.py`, `lbp.py` |
| 3 | Los 9 métodos, base de datos cifrada y defensas (ataques C, D, E) | `sistema.py`, `basedatos.py` |
| 4 | Evaluación de rendimiento y de ataques, Docker | `evaluar.py`, `ataques.py`, `Dockerfile` |

## Referencias
- T. Ahonen, A. Hadid, M. Pietikäinen. *Face Description with Local Binary Patterns: Application to Face Recognition.* IEEE TPAMI, 2006.
- X. Tan, B. Triggs. *Enhanced Local Texture Feature Sets for Face Recognition Under Difficult Lighting Conditions.* IEEE TIP, 2010.
- W. Wu et al. *YuNet: A Tiny Millisecond-level Face Detector.* Machine Intelligence Research, 2023.
