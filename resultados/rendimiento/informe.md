# Evaluación de rendimiento

- Fotos: 61 de 13 personas
- Fallos del detector: 0 (0.0 %)
- Umbral del sistema (config.py): **0.345**

## 1. Todos contra todos

- Pares genuinos: 141 (distancia media 0.320, máx 0.420)
- Pares impostores: 1689 (distancia media 0.378, mín 0.273)
- **EER = 19.46 %** con umbral 0.3461

| Punto de trabajo | Umbral | FAR | FRR |
|---|---|---|---|
| FAR ≤ 1 % | 0.3068 | 0.95 % | 60.28 % |
| FAR ≤ 5 % | 0.3209 | 4.91 % | 45.39 % |
| FAR ≤ 10 % | 0.3325 | 9.89 % | 35.46 % |
| EER | 0.3461 | 19.78 % | 19.15 % |
| **Umbral del sistema** | 0.3450 | 18.95 % | 21.28 % |

## 2. Protocolo de la práctica (1 foto inscrita por persona, el resto como intentos)

| Vía | Inscritos | FRR (verificar propio) | FAR (verificar ajeno) | Identificación correcta | Identificado como otro | No identificado |
|---|---|---|---|---|---|---|
| Con detector | 13 | 31.2 % (48) | 18.4 % (576) | 35.4 % | 16.7 % | 47.9 % |
| Sin detector (_sd) | 13 | 45.8 % (48) | 17.2 % (576) | 20.8 % | 25.0 % | 54.2 % |

## 3. Persona NO inscrita que intenta identificarse (conjunto abierto)

Se inscribe a todos menos a una persona y se identifican sus fotos (repetido con cada persona). Lo correcto es "no identificado".

- **Le da el nombre de otra persona: 32/61 (52.5 %)**
- Con 12 inscritos. Con 51 inscritos será mayor: hay más candidatos con los que confundirse.
