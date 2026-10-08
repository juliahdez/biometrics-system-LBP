# Detector de caras (Persona 1)

Detector basado en **YuNet** (`cv2.FaceDetectorYN`), una CNN ligera de OpenCV que devuelve la caja de la cara y 5 puntos clave (ojos, nariz y comisuras de la boca). Funciona en CPU, no necesita GPU y el modelo ocupa 230 KB.

## Instalación y prueba

```bash
pip install -r requirements.txt
python -m pytest -q                              # 22 pruebas + 1 limitación conocida
python detector.py foto.jpg --salida debug.jpg   # dibuja caja y puntos
```

Probado con OpenCV 4.10 y 5.0.

## Probar con la base de datos del profe

```bash
python probar_carpeta.py fotos_profe/ --salida resultados/
```

Recorre la carpeta (y sus subcarpetas) y separa las fotos en `ok/`, `varias/`, `sin_cara/` y `error/`, con la caja y los ojos dibujados, más un `informe.csv` y un resumen por pantalla. Revisad a mano `ok/` (que la caja y los ojos estén bien puestos) y todo lo que caiga en `varias/` o `sin_cara/`. Si muchas caras reales quedan en `sin_cara/`, probad con `--umbral 0.6`.

## Uso (interfaz acordada)

```python
from detector import detectar

caja, ojo_izq, ojo_der = detectar("foto.jpg")    # también acepta bytes o un array numpy
# caja = (x, y, w, h); ojos = (x, y), ojo_izq es el de la izquierda de la IMAGEN
```

Para más información (nariz, boca, confianza, ángulo de los ojos o todas las caras):

```python
from detector import DetectorCaras

det = DetectorCaras()
d = det.detectar_cara("foto.jpg")                  # Deteccion(caja, ojo_izq, ojo_der, nariz, ..., confianza)
d.angulo_ojos                                      # inclinación en grados (para alinear)
det.detectar_cara("foto.jpg", varias="error")      # rechaza fotos con varias caras
det.detectar_todas("grupo.jpg")                    # lista ordenada de mayor a menor
```

## Comportamiento

| Situación | Qué hace |
|---|---|
| Sin cara (o confianza < 0.8) | lanza `SinCaraError` |
| Varias caras | se queda con la más grande (o `VariasCarasError` con `varias="error"`) |
| Cara pegada al borde | recorta la caja a los límites de la imagen |
| Cara de menos de 24 px | se descarta |
| Imagen gigante | la reduce a 1280 px para detectar y devuelve coordenadas de la original |
| Gris, BGRA, float | se convierten a BGR uint8 |
| Fichero corrupto, vacío, < 32 px, > 40 Mpx o > 25 MB, NaN, tipo raro | lanza `ImagenInvalidaError` |

**Ojo:** los arrays numpy deben ir en **BGR** (lo que devuelve `cv2.imread`). Si usáis PIL o skimage (RGB), convertid antes con `cv2.cvtColor(img, cv2.COLOR_RGB2BGR)`.

## Limitaciones conocidas (para la memoria y la defensa frente al rival)

1. **Inclinación de la cabeza:** la cara se detecta bien hasta ±45°, pero los puntos de los ojos de YuNet **infravaloran la inclinación** a partir de ~10° (con 30° reales marcan ~3°). La alineación por ojos del preprocesado será parcial en caras muy inclinadas. Impacto: más falsos rechazos, no falsos aceptados. Posible mejora: un modelo de 68 landmarks (dlib).
2. **Fotos de fotos / pantallas:** el detector no tiene detección de vida (*liveness*); una foto impresa o en un móvil se detecta como cara real. Es una limitación de todo el sistema, no solo del detector.
3. **Caras de perfil o muy pequeñas:** con menos de 24 px o perfiles muy marcados no se detecta nada.
4. **Varias caras:** con la política por defecto un atacante podría meter su cara grande junto a otra; si preocupa, usar `varias="error"` en `inscribir`.
5. **Saltarse el detector:** los métodos `_sd` lo omiten por diseño; cualquier validación que haga el detector (tamaño, contenido) no se aplica ahí, así que el preprocesado o los métodos `_sd` deben validar su entrada.
