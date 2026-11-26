# Evaluación de seguridad (ataques)

Sistema: LBP, umbral 0.345. BD temporal con 13 personas (1ª foto de cada una); el resto de fotos se usan como material del atacante.

| # | Ataque | Resultado | Estado | Comentario |
|---|---|---|---|---|
| | **A. Presentación (entrada normal)** | | | |
| 1 | Impostor de esfuerzo cero (foto de otra persona) | aceptado 106/576 (18.4 %) | ⚠️ LIMITACIÓN CONOCIDA | Es la FAR del sistema con el umbral elegido (punto EER). LBP no separa mejor caras no controladas; bajar el umbral reduce la FAR a costa de la FRR. |
| 2 | Presentar la misma foto con la que se inscribió (fotos públicas) | aceptado 13/13 | ⚠️ LIMITACIÓN CONOCIDA | Las 50 fotos del profesor las tienen todos los grupos. Sin detección de vida (liveness) es imposible distinguir la foto de la persona. |
| 3 | Foto de inscripción degradada (desenfoque, luz, escala, JPEG 40) | aceptado 12/13 | ⚠️ LIMITACIÓN CONOCIDA | Igual que A2: una copia de la foto sirve. |
| 4 | Foto de inscripción en espejo | aceptado 13/13 | ⚠️ LIMITACIÓN CONOCIDA | El template es simétrico a propósito (mejora el reconocimiento). |
| 5 | Foto sin ninguna cara | SinCaraError | ✅ DEFENDIDO |  |
| 6 | Foto con varias caras (víctima grande + atacante pequeño) | aceptado | ⚠️ LIMITACIÓN CONOCIDA | Se usa la cara más grande; para inscribir con más rigor: detectar_cara(varias='error'). |
| | **B. Saltarse el detector (_sd)** | | | |
| 7 | Imágenes que no son caras (38: ruido, gato, cohete, ladrillo...) | verificar_sd aceptó 0/494; identificar_sd dio nombre 0/38 | ✅ DEFENDIDO | Sin detector no se comprueba que haya una cara; protege el umbral y la regla anti-lobo. |
| 8 | Imagen lisa (sin contenido) | ErrorPreprocesado | ✅ DEFENDIDO |  |
| 9 | Cara de inscripción recortada por la caja, presentada sin detector | aceptado 13/13 | ⚠️ LIMITACIÓN CONOCIDA | Es la misma foto (A2) por otra vía. La vía _sd alinea con la posición media de los ojos, así que es coherente con la vía normal. |
| 10 | Foto completa (sin recortar) por la vía _sd | aceptado 0/13 | ✅ DEFENDIDO | Sin detector, la foto entera no se parece al recorte de la cara. |
| | **C. Saltarse detector y extractor (_t)** | | | |
| 11 | Templates malformados (vacío, tamaño, NaN, inf, negativos, ceros, texto, .npy con pickle) | rechazados 10/10 | ✅ DEFENDIDO | validar_template exige 5900 valores en [0,1] y cada histograma sumando 1; .npy sin pickle. |
| 12 | Templates aleatorios válidos (fuerza bruta, 200 intentos) | verificar_t aceptó 0/200; identificar_t dio nombre 0/200 | ✅ DEFENDIDO |  |
| 13 | Template 'lobo' (media de muchas caras) | aceptado 0/13 (sin la defensa sería 13/13) | ✅ DEFENDIDO | Defensa: distancia mínima al centro de las caras y máximo de coincidencias. |
| 14 | Inscribir un template lobo con inscribir_t para robar identificaciones | InscripcionRechazadaError | ✅ DEFENDIDO |  |
| 15 | Lobo disimulado (α·lobo + (1-α)·cara real) | α=0.3: 0 %; α=0.2: 59 %; α=0.15: 76 %; α=0.1: 62 %; α=0.05: 40 %; cara real sola: 22 % | ⚠️ LIMITACIÓN CONOCIDA | Las mezclas pasan los filtros y aciertan bastante más que una cara real (hasta ~3 veces). Es el precio de bajar el margen anti-lobo a 0,12 y no aplicar la regla de coincidencias al verificar (con 0,16 y la regla, ninguna mezcla con α ≥ 0,15 entraba). Ver limitación 7 del README. |
| 16 | Template calculado de la foto pública de la víctima (el código es abierto) | aceptado 13/13 | ⚠️ LIMITACIÓN CONOCIDA | Equivale a A2: si la foto es pública, su template también. |
| 17 | Hill-climbing si el atacante VE la distancia (cambia celdas y se queda con lo que acerca) | aceptado tras 128 consultas | ⚠️ LIMITACIÓN CONOCIDA | Por eso los métodos solo devuelven sí/no y la CLI no tiene ninguna opción que muestre la distancia. Con solo sí/no no hay gradiente que seguir, y el límite de intentos (G) corta la búsqueda. |
| | **D. Leer la base de datos** | | | |
| 18 | Buscar los templates en claro dentro de bd.csv | no encontrados (cifrados) | ✅ DEFENDIDO | AES-256-GCM con nonce aleatorio por fila. |
| 19 | Usar directamente el campo cifrado como template | no es un template válido | ✅ DEFENDIDO |  |
| 20 | Descifrar con una clave adivinada | BaseDatosComprometidaError | ✅ DEFENDIDO |  |
| 21 | Ver quién está inscrito | los 13 nombres están en claro | ⚠️ LIMITACIÓN CONOCIDA | Los ids no se cifran (hacen falta para buscar). En esta práctica los nombres ya son públicos. |
| 22 | Robar bd.csv Y la clave | con la clave se descifra todo y se pueden usar con verificar_t | ⚠️ LIMITACIÓN CONOCIDA | La clave va en otro fichero (secretos/) o en la variable BIOMETRIA_CLAVE; si se roba, no hay protección. |
| | **E. Escribir en la base de datos** | | | |
| 23 | Cambiar un carácter de un template | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 24 | Intercambiar templates entre dos personas | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 25 | Renombrar una fila (mi template con el nombre de la víctima) | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 26 | Borrar una fila | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 27 | Duplicar una fila | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 28 | Reordenar las filas | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 29 | Añadir una fila con mi template cifrado con otra clave | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 30 | Sustituir el template de la víctima por el mío (otra clave) | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 31 | Vaciar la base de datos (solo cabecera y firma) | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 32 | Quitar la línea de firma | BaseDatosComprometidaError | ✅ DEFENDIDO | HMAC del fichero (última línea) + GCM por fila (con el id como dato asociado). |
| 33 | Modificar y recalcular la firma con una clave inventada | BaseDatosComprometidaError | ✅ DEFENDIDO |  |
| 34 | Rollback: restaurar una copia anterior válida (con su firma) | BaseDatosComprometidaError | ✅ DEFENDIDO | Contador de versión dentro de la firma y copiado en secretos/versiones.json: una copia antigua tiene versión menor. Si el atacante también controla secretos/, no sirve. |
| 35 | Borrar la BD entera para empezar de cero | BaseDatosComprometidaError | ✅ DEFENDIDO | El contador de secretos/versiones.json sabe que había datos. |
| 36 | Corromper la BD (denegación de servicio) | el sistema se detiene (BaseDatosComprometidaError) | ⚠️ LIMITACIÓN CONOCIDA | Se prefiere fallar cerrado: mejor no funcionar que funcionar con datos manipulados. Recuperación: restaurar la copia de seguridad. |
| 37 | Reinscribir a la víctima con mi cara (vía inscribir) | YaInscritoError | ✅ DEFENDIDO | No se permite reinscribir un id existente (tampoco cambiando mayúsculas). |
| 38 | Inscribir la foto de la víctima con otro nombre | InscripcionRechazadaError | ✅ DEFENDIDO | Regla de duplicados: distancia <= 0.15 a alguien ya inscrito. |
| 39 | Inscribir OTRA foto de la víctima con otro nombre | devolvió True | ⚠️ LIMITACIÓN CONOCIDA | Dos fotos distintas de la misma persona no se distinguen de dos personas parecidas. |
| | **F. Entradas maliciosas** | | | |
| 40 | Ids maliciosos (inyección de fórmulas CSV, comas, saltos de línea, comillas, rutas, muy largos) | rechazados 13/13 | ✅ DEFENDIDO | Solo letras, números, espacios y . _ ' - ; empieza por letra o número; máx. 64. |
| 41 | Ficheros de imagen maliciosos (vacío, texto, JPEG truncado, bomba de descompresión) | rechazados sin colgarse 4/4 | ✅ DEFENDIDO | Límite de 40 Mpx y 25 MB antes de decodificar. |
| 42 | Fichero de template enorme | ErrorTemplate | ✅ DEFENDIDO | Límite de 1 MB. |
| | **G. Encontrados atacando como el rival** | | | |
| 43 | Subir el umbral desde la línea de comandos (--umbral 5) | la opción --umbral no existe | ✅ DEFENDIDO | Antes: aceptaba a cualquiera. |
| 44 | Subir el umbral con la variable de entorno BIOMETRIA_UMBRAL=5 (impostor) | RECHAZADO | ✅ DEFENDIDO | El umbral es una constante; solo se puede bajar desde Python. |
| 45 | Subir el umbral desde Python (s.umbral = 5) | ValueError | ✅ DEFENDIDO | Solo se permiten umbrales más estrictos. |
| 46 | Usar una BD fabricada por el atacante (--bd / --clave) | la opción no existe | ✅ DEFENDIDO |  |
| 47 | Pedir la distancia para hacer hill-climbing (--detalle) | la opción no existe | ✅ DEFENDIDO |  |
| 48 | Inscribirse con un nombre que se ve igual que el de la víctima (A cirílica, ı turca, mayúsculas, guiones, dobles espacios, tildes) | rechazados 6/6 | ✅ DEFENDIDO | Solo alfabeto latino, y los ids se comparan sin mayúsculas, tildes ni separadores. No cubre I mayúscula, 1 o 0 en lugar de l u o (ver limitación 19 del README). |
| 49 | Diccionario SIN límite: probar los templates de las fotos públicas de otros famosos | entra como 12/13 víctimas (mediana 2.5 intentos) | ⚠️ LIMITACIÓN CONOCIDA | Consecuencia directa de la FAR del 19 %: con intentos ilimitados casi siempre se entra. |
| 50 | Diccionario CON el límite de intentos (3 fallos seguidos → bloqueo) | entra como 7/13 víctimas | ⚠️ LIMITACIÓN CONOCIDA | El bloqueo dura 300 s y se duplica cada vez; corta la búsqueda, pero quien acierta en los primeros intentos entra (FAR por intento ~19 %). El límite solo cuenta en verificar (no en identificar) y no garantiza el máximo con intentos simultáneos: ver limitaciones 9 y 20 del README. |
| 51 | Bloquear a la víctima a propósito (denegación de servicio) | la víctima legítima queda rechazada | ⚠️ LIMITACIÓN CONOCIDA | Precio del límite de intentos: el atacante puede bloquear una cuenta unos minutos. |
| 52 | Morphing: mezcla 50/50 de dos caras | aceptada como LAS DOS personas en 5/6 | ⚠️ LIMITACIÓN CONOCIDA | Ataque conocido en biometría facial; detectar morphs requiere modelos específicos. |
| 53 | 4 procesos inscribiendo a la vez (antes corrompía la BD para siempre) | 24/24 inscritos, BD íntegra | ✅ DEFENDIDO | Bloqueo de fichero entre procesos (bloqueo.py) en cada lectura y escritura. |
| 54 | Inscripción masiva para ralentizar el sistema | cortada al llegar al máximo (InscripcionRechazadaError) | ✅ DEFENDIDO | Máximo de 200 inscritos (con 300 filas cada operación tardaba ~0,5 s). |
| 55 | Leer ficheros ajenos con verificar_t (antes el error mostraba su primera línea) | error sin contenido del fichero; /etc/passwd → ErrorTemplate | ✅ DEFENDIDO | Solo .npy/.csv/.txt y mensajes de error genéricos. |
| | **H. Encontrados por la revisión externa** | | | |
| 56 | PNG de 642 KB que dice medir 25000x25000 (importando cv2 antes) | rechazado con un pico de 683 MB (antes 3,7 GB) | ⚠️ LIMITACIÓN CONOCIDA | Se leen las dimensiones de la cabecera (Pillow) antes de decodificar. |
| 57 | Ctrl+C / apagón a mitad de una inscripción (antes dejaba la BD inservible para siempre) | la BD sigue funcionando | ✅ DEFENDIDO | La firma va dentro del .csv: un único reemplazo atómico por escritura. |
| 58 | Subir la BD con git en Windows (core.autocrlf convierte \n en \r\n) | BD aceptada | ✅ DEFENDIDO | Se normalizan los saltos de línea antes de comprobar la firma; clave en hexadecimal; .gitattributes. |
| 59 | Usar una segunda BD con la misma clave (antes: 'falta bd.csv... ¿la han borrado?') | funcionan las dos | ✅ DEFENDIDO | Un contador de versión por BD. |
| 60 | Excepciones sin controlar: .npy con cabecera de tamaño absurdo, foto con dtype=object | controladas 2/2 | ✅ DEFENDIDO |  |
| 61 | Persona NO inscrita usa identificar (por ejemplo, alguien del grupo rival) | recibe el nombre de otro en 32/61 (52 %) | ⚠️ LIMITACIÓN CONOCIDA | Es la FAR en modo identificación, con 12 inscritos; con 51 será mayor. Limitación de LBP; se reduce con un umbral más estricto. |

**42 de 61 ataques defendidos; 19 son limitaciones conocidas y documentadas.**
