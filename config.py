"""
Configuración del sistema. Las rutas se pueden cambiar con variables de entorno.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent

# Base de datos (.csv, con su firma de integridad en la última línea)
RUTA_BD = Path(os.environ.get("BIOMETRIA_BD", RAIZ / "datos" / "bd.csv"))

# Clave maestra (32 bytes). Va en un fichero APARTE de la base de datos:
# quien solo consiga leer o escribir el .csv no puede descifrar ni falsificar nada.
# También se puede dar en hexadecimal con la variable BIOMETRIA_CLAVE.
RUTA_CLAVE = Path(os.environ.get("BIOMETRIA_CLAVE_FICHERO", RAIZ / "secretos" / "clave.key"))

# Contadores de versión de las BD (anti-rollback), uno por BD. Van junto a la clave,
# fuera de datos/.
RUTA_VERSIONES = Path(os.environ.get("BIOMETRIA_VERSIONES", RUTA_CLAVE.with_name("versiones.json")))

# Umbral de decisión (distancia chi-cuadrado). Por debajo o igual -> misma persona.
# Calibrado con 61 fotos reales de 13 famosos (evaluar.py): es el punto EER
# (FAR ~ FRR ~ 19,5 %). Más bajo = más seguro (menos impostores aceptados) pero
# más rechazos de usuarios legítimos.
# No se puede subir desde fuera (ni variable de entorno ni opción de la CLI): con
# "--umbral 5" se aceptaba a cualquiera. Solo se puede BAJAR (ser más estricto)
# con SistemaBiometrico(umbral=...).
UMBRAL = 0.345

# Al inscribir, si el nuevo template está a esta distancia o menos de alguien ya
# inscrito, se rechaza: es la misma foto (o casi) con otro nombre. Está por debajo
# de la mínima distancia entre personas distintas observada (0,273).
UMBRAL_DUPLICADO = 0.15

# Defensa contra templates "lobo": un template que es la media de muchas caras se
# parece a TODO el mundo (sin defensa, se acepta como el 100 % de los inscritos).
#  1) Se rechaza cualquier template a esta distancia o menos del "centro" de las
#     caras (media de la BD y media de referencia de 61 caras públicas). Los lobos
#     están a ~0,015-0,03. Las caras reales medidas están a 0,181 o más (0,186 si la
#     propia cara no entra en la media). Se deja 0,12 para tener margen, porque el
#     centro y estas cifras salen de las mismas 61 fotos.
#  2) Solo al INSCRIBIR: se rechaza si coincide (distancia <= UMBRAL) con más de esta
#     fracción de los inscritos. Una cara real coincide de media con el 24 % y como
#     mucho con el 42 % (60 fotos), pero con galerías pequeñas puede pasar del 50 %;
#     por eso no se aplica al verificar (rechazaría siempre a esa persona) y el
#     límite es alto.
UMBRAL_LOBO = 0.12
MAX_FRACCION_COINCIDENCIAS = 0.75
MIN_INSCRITOS_PARA_CENTRO = 10          # mínimo para usar la media de la BD como centro
MIN_INSCRITOS_PARA_COINCIDENCIAS = 20   # con menos inscritos el porcentaje es demasiado ruidoso
RUTA_CENTRO_REFERENCIA = RAIZ / "models" / "centro_referencia.npy"

# Identificación: el mejor candidato debe ganar al segundo por al menos este margen.
# Si no, el sistema responde "no identificado" en vez de arriesgarse a dar un nombre
# equivocado. Con nuestras fotos de prueba: identificaciones erróneas 48 % -> 17 %,
# correctas 48 % -> 35 %. Poner 0 para desactivarlo.
MARGEN_IDENTIFICACION = 0.01

# Límite de intentos de verificación (contra ataques de diccionario: probando fotos
# de otros famosos se entraba como 12 de 13 víctimas, con 2-3 intentos de mediana).
# Tras MAX_FALLOS_SEGUIDOS fallos seguidos contra un id, ese id queda bloqueado
# BLOQUEO_SEGUNDOS, y el tiempo se duplica en cada bloqueo nuevo. Un acierto lo reinicia.
MAX_FALLOS_SEGUIDOS = 3
BLOQUEO_SEGUNDOS = 300
RUTA_INTENTOS = Path(os.environ.get("BIOMETRIA_INTENTOS", RUTA_CLAVE.with_name("intentos.json")))

# Máximo de personas inscritas (cada operación lee la BD entera: con miles de filas
# un atacante podría ralentizar el sistema). La práctica necesita 51.
MAX_INSCRITOS = 200

# Formato de los identificadores
LONGITUD_MAX_ID = 64
