# Reconocimiento facial con LBP: detector + preprocesado + extractor

```
foto ──► detectar() ──► preprocesar() ──► extraer() ──► template ──► comparar(t1, t2) ──► distancia
         detector.py    preprocesado.py   lbp.py                     lbp.py
```

| Fichero | Qué hace |
|---|---|
| `detector.py` | Detecta la cara con **YuNet** (CNN de OpenCV, 230 KB, va en CPU) y devuelve caja y ojos |
| `preprocesado.py` | Alinea la cara por los ojos, la pasa a gris 128×128 y normaliza la luz (CLAHE) |
| `lbp.py` | Extractor LBP (8 vecinos, radio 2, rejilla 8×8 → template de 3776 valores) y distancia chi-cuadrado |
| `pipeline.py` | `obtener_template(foto)` y `obtener_template_sd(foto)` (sin detector) |
| `probar_carpeta.py` | Pasa el detector por una carpeta de fotos y separa ok / varias / sin cara / error |
| `evaluar.py` | FAR, FRR, EER, umbral recomendado y acierto en identificación sobre una carpeta |
| `ajustar.py` | Prueba varias combinaciones de parámetros LBP y dice cuál funciona mejor |

## Instalación y prueba

```bash
pip install -r requirements.txt
python -m pytest -q        # 40 pruebas + 1 limitación conocida
```

Probado con OpenCV 4.10 y 5.0, también desde rutas con `º`, tildes o `ñ`.

## Uso

```python
from pipeline import obtener_template, obtener_template_sd, comparar

t1 = obtener_template("foto1.jpg")      # ruta, bytes o array numpy BGR
t2 = obtener_template("foto2.jpg")
comparar(t1, t2)                        # 0 = idénticos ... 2 = nada en común
```

Por piezas (interfaz acordada en el grupo):

```python
from detector import detectar
from preprocesado import preprocesar
from lbp import extraer, comparar, validar_template

caja, ojo_izq, ojo_der = detectar(foto)
cara = preprocesar(foto, caja, ojo_izq, ojo_der)     # 128x128 uint8
template = extraer(cara)                              # float32, 3776 valores
validar_template(template)                            # para los métodos _t: rechaza templates manipulados
```

**Ojo:** los arrays numpy deben ir en **BGR** (lo que devuelve `cv2.imread`). Si venís de PIL o skimage (RGB), convertid con `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)`.

## Con la base de datos del profe

Organizad las fotos con **una subcarpeta por persona**:

```
fotos_profe/
  persona_01/  1.jpg  2.jpg ...
  persona_02/  ...
```

```bash
python probar_carpeta.py fotos_profe/ --salida res_detector/   # ¿detecta bien todas las caras?
python evaluar.py fotos_profe/ --salida res_eval/              # EER, umbral, identificación, gráfica
python ajustar.py fotos_profe/                                  # mejores parámetros LBP
```

Si `ajustar.py` encuentra una combinación mejor, copiad `PUNTOS`, `RADIO` y `REJILLA` en `lbp.py` y volved a pasar `evaluar.py`. El **umbral** de verificación sale de `evaluar.py` ("Umbral con FAR <= 1 %" si priorizáis seguridad, "EER" si queréis equilibrio). Si el profe da **una sola foto por persona**, no hay pares genuinos: haced fotos propias de 2–3 personas para tener genuinos.

## Cómo funciona el LBP (para la memoria)

1. Cada píxel se compara con 8 vecinos en un círculo de radio 2: vecino ≥ centro → 1, si no → 0. Los 8 bits forman un código que describe la textura local (borde, esquina, mancha...).
2. Solo se distinguen los patrones *uniformes* (como mucho 2 cambios 0↔1), que son el ~90 % de los de una cara; el resto va a una casilla común. Así salen 59 casillas.
3. La cara (alineada) se divide en una rejilla 8×8 y se hace un histograma por celda, para conservar **dónde** está cada textura.
4. El template es la concatenación de los 64 histogramas normalizados (64 × 59 = 3776 valores).
5. Dos templates se comparan con la distancia **chi-cuadrado**, la habitual para histogramas.

El LBP es invariante a cambios **globales** de brillo (solo compara píxeles entre sí); CLAHE ayuda con iluminación desigual.

## Prueba de humo (no son resultados finales)

Con 8 personas (6 de la foto de ejemplo de `insightface`, Tom Hanks y la astronauta de `scikit-image`) y 8 variantes artificiales de cada una (giro ±8°, escala, luz, desenfoque, JPEG): **EER ≈ 8 %, identificación rank-1 = 100 %**. Es optimista, porque las variantes salen de la misma foto original; los números de verdad saldrán con las fotos del profe.

## Limitaciones conocidas (para la memoria y la defensa frente al rival)

1. **Inclinación de la cabeza:** la cara se detecta bien hasta ±45°, pero los ojos de YuNet **infravaloran la inclinación** a partir de ~10° (con 30° reales marcan ~3°), así que la alineación queda incompleta en caras muy inclinadas. Impacto: más falsos rechazos, no falsos aceptados.
2. **Perfiles y caras pequeñas:** LBP depende de que la cara esté frontal y alineada; con perfiles el template cambia mucho. Caras de menos de 24 px no se detectan.
3. **Sin detección de vida:** una foto impresa o en un móvil se acepta como cara real.
4. **Métodos `_sd`:** sin detector no hay ojos, así que la cara no se alinea igual que en la inscripción. La misma foto da distancia ≈ 0.49 entre la vía normal y la `_sd`, del orden de un par genuino: funciona, pero con más error.
5. **Templates en la base de datos:** el histograma LBP no permite reconstruir la cara fácilmente, pero un atacante que lea el .csv puede **reutilizar** un template con `verificar_t` / `identificar_t`. Hay que cifrar y firmar el .csv.
6. **Templates manipulados:** `validar_template` rechaza tamaños incorrectos, NaN, negativos y histogramas sin normalizar, pero un atacante puede fabricar un template **válido** (p. ej. un promedio de muchos) e intentar que se parezca a varias personas a la vez (*wolf attack*). Conviene probarlo.
7. **Varias caras:** por defecto se usa la más grande; con `DetectorCaras().detectar_cara(foto, varias="error")` se rechazan.
8. **Dependencia del umbral:** el umbral óptimo depende de las fotos; hay que recalcularlo con la base de datos real.
