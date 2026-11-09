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

## 2026-11-09 — 005. Cambio de dinámica: resultado negativo

**Hipótesis del ajuste.** Probar si la fuga lenta del estado dificultaba aprovechar
interacciones temporales. Cambiar únicamente alpha de 0,3 a 1, manteniendo datos,
routing, semillas 7/8/9, optimizador, 500 actualizaciones y batch 64. Medido, nulo
de grados y control sin aristas reciben el mismo ajuste.

**Ejecutado.** Commit `85dd523510ee696e4ade7f743e3b1ba5529740a8`. App
`ap-VmT53KXfCtatqKuVgES0GX`, campaña `20261109T184111Z-46d08176`: los nueve ensayos
terminan sus 500 pasos en T4. Tiempo total dentro de la función: 222,58 s.
[Informes y configuración](../reports/gpu_alpha1/campaign.json).

| Modelo | Error cuadrático validación, alpha 0,3 | Error cuadrático validación, alpha 1 |
| --- | ---: | ---: |
| Conectoma medido | 1.64505 ± 0.03368 | 1.81862 ± 0.06452 |
| Sin aristas | 1.65978 ± 0.03309 | 1.81132 ± 0.04008 |
| Nulo de grados | 1.64327 ± 0.02559 | 1.81023 ± 0.05291 |

Con alpha 1, los errores en historias de final idéntico son 1,83493, 1,84479 y
1,85806, respectivamente. También empeoran respecto al régimen original.

**Decisión.** Conservar alpha 0,3 como valor por defecto. No hay evidencia de que
el ajuste propuesto resuelva el aprendizaje de movimiento en este régimen. La
opción permanece disponible para reproducir el resultado, sin presentarla como
una mejora. No aumentar escala o presupuesto solo por preferencia por el modelo.

**Reflexión sobre ingeniería.** Se exploraron localmente operaciones alternativas
para el cálculo recurrente. Una multiplicación dispersa fue más lenta que la
acumulación por arista en el microbenchmark; una matriz densa enmascarada fue más
rápida, pero requiere memoria cuadrática y el resultado no es un benchmark de
entrenamiento completo. No se cambió el núcleo por esa observación aislada.

**Estado al cerrar esta primera ronda.** 33 ensayos GPU completos (12 referencias,
12 MICrONS originales, 9 con alpha 1), además de 16 ensayos comparativos CPU
conservados y las pruebas de integración. 27 tests pasan. Los cuatro intentos de
app de BIONN figuran detenidos, con cero tareas activas. No queda una campaña GPU
ejecutándose ni se ha programado una automatización recurrente.

La consulta de facturación registra 0,11488725 USD en ejecuciones efímeras del
workspace; es una fotografía provisional, no una factura final atribuida al
proyecto. Se conservan 8 USD de reservas operativas para cuatro intentos y 42 USD
de margen frente al máximo autorizado de 50 USD. No reiniciar ese presupuesto en
la siguiente sesión.

**Próximos experimentos propuestos, todavía no ejecutados.**

1. Comprobar sobreajuste deliberado de un conjunto muy pequeño y medir el error
   completo de entrenamiento para separar capacidad de optimización.
2. Revisar escalas de gradientes, ganancia recurrente y representación sensorial,
   manteniendo controles emparejados. Reservar conjuntos nuevos para confirmar
   cualquier decisión tomada con esta validación.
3. Solo después, comparar nulos que conserven tipo y distancia y calibrar una
   interfaz de entrada más defendible. No reclamar especificidad anatómica con el
   nulo de grados actual ni generalización a otros animales.

## Entrada 006 — 2026-11-09: diagnóstico de capacidad con 32 ejemplos (plan previo)

**Motivación.** Las tres semillas del conectoma a 500 actualizaciones no mejoran
la predicción cero en validación; quitar la fuga (`alpha=1`) empeora el resultado.
No repetir ese ajuste sin evidencia nueva. Antes de buscar una ventaja anatómica,
comprobar que el modelo puede ajustar un conjunto deliberadamente pequeño.

**Hipótesis y criterio.** Si conectoma y control sin aristas reducen el MSE de
entrenamiento completo por debajo del 1 % de su referencia cero, tienen capacidad
para memorizar 32 clips bajo este optimizador. Esto no demuestra capacidad para
la tarea general ni que 500 pasos basten con 2048 clips. Si fallan mientras GRU
aprende, investigar representación/gradientes antes de ampliar el estudio del grafo.
La lectura estática controla que memorizar ejemplos no equivale a usar movimiento.

**Diseño fijado antes de lanzar.** MICrONS v1718, 32 clips de entrenamiento,
512 de validación y las mismas 16 texturas contrafactuales × 8 direcciones; filtros
4×4, 8 fotogramas, alpha 0,3. Modelos GRU, estático, conectoma y sin aristas,
semillas 7/8/9, 1500 actualizaciones Adam, lr 0,003, batch 64. Datos y calibración
compartidos dentro de esta campaña. Cambiar el tamaño del generador cambia las
texturas, etiquetas y calibración: no asumir que estos 32 clips son un subconjunto
de los 2048 anteriores ni comparar directamente ambos errores de validación.
Medir todo el entrenamiento después del último paso, además de validación e
historias contrafactuales. Registrar normas de gradiente y cambio de parámetros.

**Ejecución y coste.** Una campaña T4 secuencial de 12 ensayos, límite suave
1200 s y duro 1800 s, sin reintentos; reserva previa de 2 USD. No hay aplicaciones
BIONN activas al preparar este plan. El contador agregado de aplicaciones efímeras
de Modal es 0,10324261 USD, no una factura final ni un desglose exclusivo del
proyecto; se mantienen todas las reservas anteriores. Resultados pendientes.

**Continuidad nocturna.** Por petición del usuario se ha creado una revisión en
este chat cada dos horas durante la noche del 9 al 10 de noviembre, con límite
el 10 a las 08:00 Europe/Paris. Consultará estado, bitácora y presupuesto antes de
actuar; no duplicará campañas activas ni repetirá hipótesis descartadas sin motivo.
El máximo sigue siendo 50 USD acumulados para toda la ronda.

### Diagnóstico adicional fijado durante la campaña 006

La reevaluación en CPU de nueve checkpoints de la campaña original confirma MSE
medio de entrenamiento: GRU 0,044127, conectoma 1,389131 y sin aristas 1,422518.
Se reconstruyeron exactamente datos/calibración y se reprodujo la validación
con diferencias numéricas inferiores a la tolerancia fijada (relativa 1e-5,
absoluta 1e-6). No hubo entrenamiento adicional. El grafo local tiene el mismo
SHA-256 que el usado en Modal. Archivo: `reports/retrospective_training.json`.

El primer ensayo pequeño del conectoma baja a MSE 0,013758 en entrenamiento
pero sube a 3,022269 en validación; es parcial, no una conclusión de tres semillas.
Esto motiva una prueba independiente **sin gasto GPU**: congelar los núcleos ya
entrenados de la campaña original (2048 clips, 500 pasos) y resolver directamente
la lectura lineal mediante mínimos cuadrados/ridge. Comparar conectoma y sin
aristas, semillas 7/8/9, con penalizaciones fijadas ahora: 0, 1e-4, 1e-2, 1 y 100.
Centrar y escalar cada estado final usando exclusivamente entrenamiento; el sesgo
no se penaliza. Las penalizaciones se aplican al error cuadrático medio por clip,
no a su suma. Resolver en float64 y registrar rango/tolerancia numérica.

Si solo mejora entrenamiento, hay margen de optimización de la lectura pero no
se ha mostrado una representación que generalice. Si mejora validación, comparar
ambos controles con el mismo presupuesto de selección y confirmarlo después
con particiones nuevas. La selección entre penalizaciones sigue siendo ajuste
sobre validación; no es un test final ni evidencia aislada de ventaja anatómica.
No modificar el núcleo ni descartar puntos de regularización desfavorables.

### Resultado de la campaña 006

Completados **12/12 ensayos de 1500 pasos**, T4, 224,42 s de función. Revisión
`3f183683d7d696adb41129983396b5ad89801949`; aplicación
`ap-jTYXYN8vV4Zu4BraLAilMz`, campaña `20261109T190712Z-eb2cacc2`.
Reportes íntegros en `reports/gpu_tiny_fit/`; checkpoints privados recuperados.
Ninguna aplicación BIONN permanece activa al comprobar el cierre.

| Modelo | MSE entrenamiento, media ± DE | MSE validación, media ± DE | MSE contrafactual, media ± DE |
| --- | ---: | ---: | ---: |
| GRU | 3,12e-13 ± 1,81e-14 | 2,10916 ± 0,02565 | 2,03630 ± 0,02296 |
| Estático | 0,02115 ± 0,00285 | 10,34310 ± 0,43679 | 11,84951 ± 0,28159 |
| Conectoma | 0,00459 ± 0,00794 | 2,98154 ± 0,08968 | 2,38177 ± 0,02737 |
| Sin aristas | 0,00648 ± 0,00557 | 2,98249 ± 0,06466 | 2,37420 ± 0,04162 |
| Velocidad cero | 1,43750 | 1,49805 | 1,50000 |

**Observación.** Conectoma y sin aristas cumplen el criterio fijado en las tres
semillas: MSE de entrenamiento <0,014375 (1 % de cero). La semilla 7 del conectoma
queda cerca del umbral: 0,013758; las otras dan 2,96e-8 y 4,09e-6. Su norma de
gradiente al último lote aún es 3,71 antes del clipping, frente a 0,0063 y 0,0220
en las otras dos. No se declara convergencia uniforme ni se oculta la variación.
Los parámetros recurrentes cambian: RMS del cambio de theta 0,0755–0,0782.

**Interpretación y decisión.** El pipeline puede memorizar estos 32 ejemplos;
queda descartada una incapacidad absoluta de ajuste bajo esta configuración.
Todos los modelos generalizan mal con tan pocos clips. La casi igualdad entre
conectoma y sin aristas no aporta una ventaja anatómica. Tampoco se debe atribuir
el fallo del conjunto grande únicamente a generalización: su error de
entrenamiento ya era alto. Completar el diagnóstico de lectura congelada antes
de elegir otro entrenamiento; no repetir el ajuste pequeño ni alpha=1.

**Gasto.** Reservas acumuladas 10 USD de los 50 autorizados; margen conservador
40 USD. La lectura de uso agregado de aplicaciones efímeras de Modal tras esta
campaña es 0,15065402 USD, todavía sujeta a retrasos y sin constituir una factura
exclusiva del proyecto. No se liberan reservas basándose en esa lectura parcial.

## Entrada 007 — 2026-11-09: resolver la lectura no rescata la representación

Se completó en CPU el diagnóstico fijado durante la entrada 006: seis núcleos
congelados (conectoma/sin aristas × tres semillas), cinco penalizaciones cada uno,
**30 lecturas ajustadas sin reentrenar el núcleo**. Herramienta reproducible:
`scripts/readout_diagnostic.py`; datos completos en `reports/readout_diagnostic/`.
Se verificaron hashes del grafo, datos/calibración y reproducción de la validación
original. Todos los estados tienen rango numérico 1357 bajo el umbral documentado.
La solución se comprobó contra un sistema lineal directo y mínimos cuadrados con
un diseño singular; los datos de validación no participan en la calibración.

| Lectura | Conectoma: MSE entrenamiento / validación | Sin aristas: MSE entrenamiento / validación |
| --- | ---: | ---: |
| Adam original | 1,38913 / 1,64505 | 1,42252 / 1,65978 |
| Mínimos cuadrados, lambda 0 | 0,44513 / 4,23610 | 0,48777 / 4,69867 |
| Ridge, lambda 0,0001 | 0,44645 / 4,02719 | 0,49204 / 4,41582 |
| Ridge, lambda 0,01 | 0,64246 / 2,16576 | 0,71069 / 2,21769 |
| Ridge, lambda 1 | 1,29518 / 1,53269 | 1,32140 / 1,53469 |
| Ridge, lambda 100 | 1,49078 / 1,49738 | 1,49144 / 1,49743 |
| Constante: media de etiquetas de entrenamiento | 1,50434 / 1,49741 | 1,50434 / 1,49741 |

Son medias de tres semillas; desviaciones y errores contrafactuales están en el
resumen JSON. El mejor lambda del conjunto probado es el extremo superior (100),
con error prácticamente idéntico a la predicción constante. En contrafactuales,
ese lambda da 1,50723/1,50752 frente a 1,50006 de la constante.

**Conclusión acotada.** Hay margen de optimización de la lectura para bajar el
error de entrenamiento, pero la lectura resuelta directamente sobre estos estados
no aporta aprendizaje temporal útil que generalice. No basta con sustituir Adam
por ridge en la salida. No se ha demostrado que optimizar conjuntamente durante
más tiempo sea inútil, ni se ha evaluado toda regularización posible. La comparación
sigue siendo exploratoria, con la validación utilizada para ajustar.

**Aclaración del control.** El modelo sin aristas es aditivo entre fotogramas,
aunque conserve estado por fuga. Esa propiedad se ha demostrado a partir de la
ecuación y comprobado mediante combinación de historias en un test. Con las
marginales estacionarias del generador, memorizar una muestra finita no le da una
señal poblacional de velocidad. El protocolo ahora distingue esta limitación de
la capacidad de interacción temporal del conectoma.

**Siguiente decisión, sin ejecutar aún.** Una sola comparación emparejada de
presupuesto de entrenamiento: 3000 actualizaciones frente a las 500 originales,
manteniendo datos (2048 clips), lr 0,003, alpha 0,3, batch 64, sensores y modelos
conectoma/sin aristas/nulo de grados, semillas 7/8/9. No cambiar a la vez interfaz,
optimizador ni dinámica. Registrar curvas y error completo de entrenamiento.
Reservar 2 USD y comprobar uso/aplicaciones antes de lanzar la campaña de nueve
ensayos (1500 s cooperativos, 1800 s de servidor). El registro actual todavía
no incluye esa reserva ni afirma que la campaña haya comenzado.

Si entrenamiento baja pero validación permanece en la constante o peor, estudiar
representación/interfaz en lugar de añadir más pasos. Si los dos errores permanecen
altos, auditar gradientes por grupo y cambios efectivos de pesos antes de otra GPU.
Si aparece una mejora consistente, contrastarla con el nulo emparejado y reservar
una partición nueva para confirmación. No volver a la prueba de 32 clips, alpha=1
ni a una búsqueda de lectura congelada sin una razón nueva registrada.
