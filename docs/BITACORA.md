# Bitácora de BIONN / V1NN

Registro acumulativo de experimentos, resultados, hipótesis, retoques y decisiones.
Las reflexiones son conclusiones técnicas resumidas con evidencia y limitaciones.
Un resultado negativo se conserva; una hipótesis no se presenta como observación.

## Cómo mantenerla

Cada entrada debe indicar fecha, objetivo, revisión del código, datos y semillas,
comandos/configuración, resultados, interpretación, límites y próximo paso.
Enlazar los JSON de métricas y distinguir resultados ejecutados de trabajo propuesto.
No guardar credenciales, datos crudos o pesos en GitHub. Añadir correcciones con fecha,
sin borrar resultados anteriores para mejorar retrospectivamente la narrativa.

## 2026-11-09 — 001. Arranque autónomo y reproducción

**Objetivo.** Convertir el prototipo existente en un piloto reproducible que permita
iterar con GPU y conservar el historial del trabajo.

**Punto de partida.** Commit `ad5468fadef647999d0ed23c9c3dce832ac049be`.
La pregunta científica sigue siendo la especificidad del cableado de V1; no se
cambia a JEPA ni se introduce un encoder preentrenado.

**Ejecutado.** Los 20 tests originales pasan. Los smoke tests de 25 actualizaciones
reproducen los errores anteriores en Python 3.12.3 / PyTorch 2.14.1+cu130, usando CPU:

| Objetivo | Error en datos nuevos | Referencia |
| --- | ---: | ---: |
| Movimiento | 1.57416689 | Velocidad cero: 1.53125000 |
| SSL sensorial | 0.00257047 | Persistencia: 0.00060486 |

**Reflexión.** Una ejecución sin errores no demuestra aprendizaje útil. Antes de
comparar conectomas hay que establecer una tarea temporal resoluble y controles
que impidan explicar el resultado por una imagen aislada.

**Acceso y recursos.** Modal conectado y token verificado; credenciales fuera del
repositorio con permisos locales 0600. Máximo autorizado para la primera ronda:
50 USD. Ningún coste GPU se había generado al redactar esta entrada. El límite
económico se lleva como presupuesto operativo; no se ha cambiado la facturación
global de la cuenta. Cada campaña tiene límites explícitos de tiempo y concurrencia.

## 2026-11-09 — 002. Importación real de MICrONS

**Ejecutado.** `python scripts/prepare_microns.py`, descarga de las dos fuentes
fijadas por el importador, comprobación de hashes de blobs y manifiesto SHA256.
Informe: [microns_import.json](../reports/microns_import.json).

| Propiedad | Valor |
| --- | ---: |
| Neuronas | 1.357 |
| Parejas dirigidas | 78.953 |
| Sinapsis incluidas | 146.711 |
| Excitadoras / inhibidoras | 1.191 / 166 |
| Signos desconocidos | 0 |
| Autapsis / neuronas aisladas | 0 / 0 |

Hay posiciones transformadas y tipos generales; no hay metadatos de capa. No
llamar «preservador de capas» a los nulos de este conjunto. La retención de
conexiones de frontera no puede estimarse con esta exportación interna. Importar
el grafo no completa la auditoría fisiológica, de licencia ni de correspondencias
funcionales. Los datos crudos y el NPZ permanecen fuera de GitHub.

## 2026-11-09 — 003. Piloto temporal y primer resultado

**Retoques.** Se añaden `scripts/pilot.py` y `v1nn.benchmark`: particiones
deterministas de entrenamiento y validación, hashes, calibración fija calculada
solo con entrenamiento, semillas, métricas de error vectorial, checkpoints y
controles estático, historia lineal, GRU, CNN temporal, conectoma, sin aristas,
reinicio de estado, reservorio y nulo que conserva grados dirigidos.

La CNN temporal opera sobre los mismos sensores fijos. Una referencia analítica
de correlación de píxeles comprueba por separado que la tarea tiene solución;
no es una comparación de arquitectura igualada. El nulo de grados aún no conserva
tipo y distancia. El generador usa velocidades conocidas y texturas nuevas;
todavía no evalúa generalización a velocidades fuera del entrenamiento.

**Prueba contra atajos.** Para cada una de 16 texturas se construyen ocho historias
con movimientos diferentes y exactamente el mismo fotograma final. Las etiquetas
están balanceadas. Un predictor estático no puede identificar el movimiento de
cada historia a partir de esa imagen común.

**Ejecutado.** GRU, semilla 7, 500 actualizaciones, 2.048 clips de entrenamiento,
512 de validación, 8 fotogramas, grid 4, 64 unidades, CPU:

| Partición | Error cuadrático vectorial | Velocidad cero |
| --- | ---: | ---: |
| Validación de desarrollo | 0.44454679 | 1.49804688 |
| Historias con final idéntico | 0.42165637 | 1.50000000 |

Tiempo del runner: 15,86 s (excluye arranque de Python). Informe:
[pilot_gru_cpu_seed7.json](../reports/pilot_gru_cpu_seed7.json).

**Interpretación.** Una referencia temporal aprende con esta interfaz. Es una
prueba de viabilidad de la tarea, no evidencia de ventaja biológica. Una semilla
y validación de desarrollo no son un resultado confirmatorio. No se usa un test
final para seleccionar hiperparámetros.

**Verificación del software.** 26 tests pasan, incluidos controles de historia,
separación de particiones, calibración sin filtración, repetibilidad y checkpoint.
La definición de Modal carga sin lanzar GPU. Ejecución remota pendiente al cerrar
esta entrada; se añadirá su resultado en una entrada posterior.

**Siguiente paso.** Medir coste/rendimiento real de GPU y comparar varias semillas
con controles emparejados. No escalar por reducir únicamente la pérdida de
entrenamiento. Mantener el techo de gasto acumulado y registrar fallos completos.

## 2026-11-09 — 004. Modal operativo y 24 ensayos completos en T4

**Incidencia conservada.** El primer arranque creó la imagen, pero Modal rechazó
la T4 por ausencia de método de pago. App `ap-5KveeqwTQlpsF6VClkQAtI`, detenida
sin lanzar GPU. El usuario añadió el método de pago y el siguiente intento funcionó.

**Revisión ejecutada.** `517e62448524e457e51d2536cd57d3c2d9909d25`, árbol limpio.
Python 3.12.10, PyTorch 2.14.1+cu130, NumPy 2.5.4, CUDA 13.0, Tesla T4.
La identidad del commit remoto está en `campaign.json`; los informes individuales
registran además el hash de los archivos Python. Checkpoints recuperados localmente
y conservados en Modal; no se publican en GitHub.

**Referencias, tres semillas (7/8/9), 500 pasos, batch 64.** Mismos 2.048 clips
de entrenamiento y 512 de validación. Misma calibración de sensores para todas.
Los valores ± son desviaciones estándar entre entrenamientos, no intervalos de
confianza, ni variabilidad entre animales. EPE es error euclídeo de velocidad.

| Modelo | Error cuadrático validación | EPE validación | Error cuadrático, final idéntico |
| --- | ---: | ---: | ---: |
| GRU | 0.43005 ± 0.01903 | 0.56185 | 0.42689 |
| CNN temporal | 0.64233 ± 0.01549 | 0.71636 | 0.64539 |
| Estático | 1.52809 ± 0.01459 | 1.21212 | 1.53245 |
| Historia lineal | 1.79634 ± 0.01880 | 1.27127 | 1.72191 |
| Velocidad cero | 1.49805 | 1.20630 | 1.50000 |

Informes: [GPU](../reports/gpu_reference/campaign.json),
[reproducción CPU](../reports/cpu_reference/summary.json).
La campaña GPU duró 52,25 s dentro de la función; el total de los runners CPU
fue 54,69 s. No son tiempos completos de facturación ni un benchmark de hardware
aislado; incluyen preparaciones y, en la campaña, commits al volumen.

**MICrONS, mismos datos y tres semillas, 500 pasos, batch 64, alpha 0,3.**

| Modelo | Error cuadrático validación | EPE validación | Error cuadrático, final idéntico |
| --- | ---: | ---: | ---: |
| Conectoma medido | 1.64505 ± 0.03368 | 1.23104 | 1.70956 |
| Sin aristas | 1.65978 ± 0.03309 | 1.23637 | 1.72310 |
| Estado reiniciado | 1.61528 ± 0.03641 | 1.23383 | 1.65848 |
| Nulo de grados dirigidos | 1.64327 ± 0.02559 | 1.23067 | 1.70920 |

[Campaña MICrONS](../reports/gpu_microns/campaign.json): 12 ensayos completos,
191,53 s dentro de la función. No hay mejora frente a velocidad cero. Medido y
nulo quedan prácticamente iguales en este régimen. No se interpreta como
refutación general de los conectomas: interfaz, dinámica y optimización son
limitaciones sustanciales. Este nulo conserva grados, **no tipos ni distancias**.

**Exploración previa de CPU.** Se conservan [cuatro ensayos](../reports/cpu_microns_short/summary.json)
de la semilla 7, 100 pasos y batch 16. Se detuvo la continuación de esa campaña
cuando estuvo disponible la GPU; no se presentan las otras semillas como ejecutadas.
El nulo realizó 394.765 intercambios en 574.595 intentos y cambió el 83,95 % de las
aristas. Esto mide cambio efectivo, no mezcla uniforme. Auditoría estructural:
[microns_audit.json](../reports/microns_audit.json).

**Reflexión y siguiente retoque.** La GRU reduce mucho su pérdida de entrenamiento;
MICrONS con alpha 0,3 permanece cerca del nivel trivial. Antes de aumentar tamaño
o presupuesto, se expone `--alpha` para probar si una actualización más rápida
facilita interacciones temporales. Mantener los mismos datos, signos, routing,
topología y presupuesto entre medido, nulo y control sin aristas. Este ajuste es
exploratorio sobre validación; no modifica retrospectivamente el protocolo final.

**Comprobaciones y gasto.** 27 tests pasan, incluida la propagación de alpha y su
efecto sobre la memoria individual. Se consultó facturación antes y después; las
cifras inmediatas pueden llegar con retraso. Se mantienen 6 USD de reservas
conservadoras para los tres intentos, no 6 USD de gasto medido. El techo sigue
siendo 50 USD acumulados. [Registro de cómputo](../reports/compute_ledger.json).
