"""
Detector de caras - Práctica Sistemas Biométricos (aproximación LBP).

Usa YuNet (OpenCV, cv2.FaceDetectorYN): detector CNN ligero que devuelve,
además de la caja de la cara, 5 puntos clave (ojos, nariz y comisuras de la boca).

Interfaz acordada con el grupo:
    detectar(foto) -> (caja, ojo_izq, ojo_der)

    caja    = (x, y, w, h) en píxeles, ya recortada a los límites de la imagen
    ojo_izq = (x, y) del ojo que aparece a la IZQUIERDA en la imagen
    ojo_der = (x, y) del ojo que aparece a la DERECHA en la imagen

`foto` puede ser una ruta (str/Path), los bytes de un fichero de imagen o un
array de numpy (BGR, RGB no: OpenCV trabaja en BGR; escala de grises y BGRA
también se aceptan y se convierten).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
RUTA_MODELO = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"

UMBRAL_CONFIANZA = 0.8      # detecciones por debajo se descartan
UMBRAL_NMS = 0.3            # solapamiento máximo entre detecciones
LADO_MAX_DETECCION = 1280   # imágenes más grandes se reducen para detectar
TAM_MIN_IMAGEN = 32         # lado mínimo (px) aceptado de la imagen
TAM_MIN_CARA = 24           # lado mínimo (px) de una cara válida
MAX_PIXELES = 40_000_000    # protección contra imágenes gigantes (DoS)
MAX_BYTES = 25 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Errores
# --------------------------------------------------------------------------- #
class ErrorDetector(Exception):
    """Error base del detector."""


class ImagenInvalidaError(ErrorDetector):
    """La entrada no es una imagen válida (no existe, corrupta, tamaño absurdo...)."""


class SinCaraError(ErrorDetector):
    """No se ha encontrado ninguna cara con suficiente confianza."""


class VariasCarasError(ErrorDetector):
    """Hay varias caras y la política elegida es rechazar."""


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Deteccion:
    caja: tuple[int, int, int, int]          # (x, y, w, h)
    ojo_izq: tuple[float, float]             # ojo a la izquierda de la imagen
    ojo_der: tuple[float, float]             # ojo a la derecha de la imagen
    nariz: tuple[float, float]
    boca_izq: tuple[float, float]
    boca_der: tuple[float, float]
    confianza: float

    @property
    def area(self) -> int:
        return self.caja[2] * self.caja[3]

    @property
    def angulo_ojos(self) -> float:
        """Inclinación de la línea de los ojos en grados (0 = horizontal)."""
        dx = self.ojo_der[0] - self.ojo_izq[0]
        dy = self.ojo_der[1] - self.ojo_izq[1]
        return float(np.degrees(np.arctan2(dy, dx)))


# --------------------------------------------------------------------------- #
# Carga y validación de la imagen
# --------------------------------------------------------------------------- #
def cargar_imagen(foto) -> np.ndarray:
    """Devuelve la imagen como array BGR uint8 de 3 canales, o lanza ImagenInvalidaError."""
    if isinstance(foto, (str, os.PathLike)):
        ruta = Path(foto)
        if not ruta.is_file():
            raise ImagenInvalidaError(f"No existe el fichero: {ruta}")
        if ruta.stat().st_size > MAX_BYTES:
            raise ImagenInvalidaError("Fichero demasiado grande")
        foto = ruta.read_bytes()

    if isinstance(foto, (bytes, bytearray, memoryview)):
        if len(foto) == 0 or len(foto) > MAX_BYTES:
            raise ImagenInvalidaError("Tamaño de fichero no válido")
        buf = np.frombuffer(bytes(foto), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ImagenInvalidaError("No se pudo decodificar la imagen (formato no soportado o corrupta)")
    elif isinstance(foto, np.ndarray):
        img = foto
    else:
        raise ImagenInvalidaError(f"Tipo de entrada no soportado: {type(foto).__name__}")

    if img.size == 0 or img.ndim not in (2, 3):
        raise ImagenInvalidaError(f"Forma de imagen no válida: {img.shape}")
    h, w = img.shape[:2]
    if h * w > MAX_PIXELES:
        raise ImagenInvalidaError(f"Imagen demasiado grande: {w}x{h}")
    if min(h, w) < TAM_MIN_IMAGEN:
        raise ImagenInvalidaError(f"Imagen demasiado pequeña: {w}x{h}")

    # Normalizar tipo de dato a uint8
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            if not np.isfinite(img).all():
                raise ImagenInvalidaError("La imagen contiene NaN o infinitos")
            img = np.clip(img * 255.0 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)
        else:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Normalizar canales a BGR
    if img.ndim == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] != 3:
        raise ImagenInvalidaError(f"Número de canales no soportado: {img.shape[2]}")

    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
class DetectorCaras:
    def __init__(
        self,
        ruta_modelo: str | os.PathLike = RUTA_MODELO,
        umbral_confianza: float = UMBRAL_CONFIANZA,
        umbral_nms: float = UMBRAL_NMS,
        tam_min_cara: int = TAM_MIN_CARA,
    ):
        ruta_modelo = Path(ruta_modelo)
        if not ruta_modelo.is_file():
            raise FileNotFoundError(f"No se encuentra el modelo YuNet en {ruta_modelo}")
        # Se lee el modelo desde Python y se pasa en memoria: OpenCV en Windows
        # no abre rutas con caracteres no ASCII (º, tildes, ñ...).
        modelo = np.frombuffer(ruta_modelo.read_bytes(), dtype=np.uint8)
        if modelo.size < 100_000:
            raise ValueError(
                f"El modelo {ruta_modelo.name} pesa {modelo.size} bytes y debería pesar ~230 KB: "
                "está corrupto o no se descargó bien (¿OneDrive solo en la nube?)"
            )
        self.umbral_confianza = umbral_confianza
        self.tam_min_cara = tam_min_cara
        self._detector = cv2.FaceDetectorYN.create(
            "onnx", modelo, np.empty(0, np.uint8), (320, 320),
            umbral_confianza, umbral_nms, 5000
        )

    def detectar_todas(self, foto) -> list[Deteccion]:
        """Todas las caras válidas de la imagen, ordenadas de mayor a menor."""
        img = cargar_imagen(foto)
        h, w = img.shape[:2]

        # Reducir imágenes muy grandes: YuNet funciona mejor y más rápido así
        escala = min(1.0, LADO_MAX_DETECCION / max(h, w))
        img_det = img if escala == 1.0 else cv2.resize(
            img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA
        )

        self._detector.setInputSize((img_det.shape[1], img_det.shape[0]))
        _, caras = self._detector.detect(img_det)
        if caras is None:
            return []

        detecciones = []
        for f in caras:
            f = f.astype(np.float64)
            f[:14] /= escala                      # volver a coordenadas originales
            x, y, cw, ch = f[0:4]
            confianza = float(f[14])
            if confianza < self.umbral_confianza:
                continue

            # Recortar la caja a los límites de la imagen
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(w, int(round(x + cw))), min(h, int(round(y + ch)))
            if min(x1 - x0, y1 - y0) < self.tam_min_cara:
                continue

            p = [(float(f[i]), float(f[i + 1])) for i in range(4, 14, 2)]
            ojos = sorted(p[0:2], key=lambda q: q[0])
            boca = sorted(p[3:5], key=lambda q: q[0])
            detecciones.append(Deteccion(
                caja=(x0, y0, x1 - x0, y1 - y0),
                ojo_izq=ojos[0], ojo_der=ojos[1], nariz=p[2],
                boca_izq=boca[0], boca_der=boca[1],
                confianza=confianza,
            ))

        detecciones.sort(key=lambda d: d.area, reverse=True)
        return detecciones

    def detectar_cara(self, foto, varias: str = "mayor") -> Deteccion:
        """
        Devuelve una única cara.
        varias="mayor": si hay varias, se queda con la más grande (por defecto).
        varias="error": si hay varias, lanza VariasCarasError (más estricto).
        """
        if varias not in ("mayor", "error"):
            raise ValueError("varias debe ser 'mayor' o 'error'")
        detecciones = self.detectar_todas(foto)
        if not detecciones:
            raise SinCaraError("No se ha detectado ninguna cara")
        if varias == "error" and len(detecciones) > 1:
            raise VariasCarasError(f"Se han detectado {len(detecciones)} caras")
        return detecciones[0]


# --------------------------------------------------------------------------- #
# Interfaz del grupo
# --------------------------------------------------------------------------- #
_detector_global: DetectorCaras | None = None


def _get_detector() -> DetectorCaras:
    global _detector_global
    if _detector_global is None:
        _detector_global = DetectorCaras()
    return _detector_global


def detectar(foto):
    """detectar(foto) -> (caja, ojo_izq, ojo_der). Lanza SinCaraError / ImagenInvalidaError."""
    d = _get_detector().detectar_cara(foto)
    return d.caja, d.ojo_izq, d.ojo_der


def guardar_imagen(ruta, img: np.ndarray) -> None:
    """Como cv2.imwrite, pero funciona con rutas con º, tildes o ñ en Windows."""
    ruta = Path(ruta)
    ok, buf = cv2.imencode(ruta.suffix or ".jpg", img)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como {ruta.suffix}")
    ruta.write_bytes(buf.tobytes())


def dibujar(foto, detecciones: list[Deteccion] | Deteccion) -> np.ndarray:
    """Imagen BGR con las cajas y puntos dibujados, para depurar."""
    img = cargar_imagen(foto).copy()
    if isinstance(detecciones, Deteccion):
        detecciones = [detecciones]
    for i, d in enumerate(detecciones):
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        x, y, w, h = d.caja
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        for (px, py), c in [(d.ojo_izq, (255, 0, 0)), (d.ojo_der, (0, 0, 255)),
                            (d.nariz, (0, 255, 255)), (d.boca_izq, (255, 0, 255)),
                            (d.boca_der, (255, 0, 255))]:
            cv2.circle(img, (int(px), int(py)), 3, c, -1)
        cv2.putText(img, f"{d.confianza:.2f}", (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Detecta caras en una imagen")
    ap.add_argument("foto")
    ap.add_argument("--salida", help="guardar imagen con las detecciones dibujadas")
    args = ap.parse_args()

    det = DetectorCaras()
    caras = det.detectar_todas(args.foto)
    print(f"{len(caras)} cara(s) detectada(s)")
    for i, c in enumerate(caras):
        print(f"  [{i}] caja={c.caja} conf={c.confianza:.3f} "
              f"ojo_izq=({c.ojo_izq[0]:.0f},{c.ojo_izq[1]:.0f}) "
              f"ojo_der=({c.ojo_der[0]:.0f},{c.ojo_der[1]:.0f}) angulo={c.angulo_ojos:.1f}°")
    if args.salida:
        guardar_imagen(args.salida, dibujar(args.foto, caras))
        print(f"Guardada en {args.salida}")
