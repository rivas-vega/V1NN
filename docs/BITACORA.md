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
