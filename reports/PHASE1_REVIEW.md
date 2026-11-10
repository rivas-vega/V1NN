# BIONN / V1NN: revisión del piloto inicial

Cierre del piloto comparativo el 10 de noviembre de 2026. El objetivo fue estimar
qué aporta el cableado medido de una columna de V1 dentro de una RNN controlada.
**No se demuestra una ventaja robusta del cableado fino frente al nulo que conserva
grados E/I y organización espacial discretizada.** Esto limita la hipótesis en
este régimen; no establece equivalencia ni una conclusión general sobre cerebros.

## Evidencia que sostiene la decisión

| Pregunta | Resultado | Alcance |
| --- | --- | --- |
| ¿La tarea contiene señal temporal aprendible? | GRU y CNN temporal aprenden; controles estáticos no resuelven la tarea. | Texturas sintéticas, ocho velocidades; no escenas naturales ni extrapolación de velocidad. |
| ¿La RNN puede optimizarse y usar recurrencia? | Memoriza 32 clips; aprende mejor a 3000 pasos. Retirar corriente recurrente de checkpoints aumenta EPE +0,371 de media. | Capacidad de ajuste y dependencia del predictor; no especificidad anatómica. |
| ¿Supera al nulo de grados? | Confirmación congelada: diferencia EPE −0,02004 en 2048 clips nuevos. | Tres semillas, estos modelos y este control; no incluye nulos E/I o espaciales posteriores. |
| ¿Persiste con E/I y espacio? | Matriz 3×3: EPE medido 1,088736, nulo 1,090932; −0,20 % de media, signos mixtos. MSE medido 0,24 % peor. | Validación reutilizada, una columna; comparación cerrada sin ventaja robusta. |
| ¿La interfaz elimina dimensiones? | El routing lineal tiene rango 32. Tras ReLU aparecen pérdidas locales ocasionales por características sin ruta activa. | 0,78–1,95 % de primeras imágenes de validación por ruta en modelos aprendidos; no demuestra pérdida de movimiento. |

Fuentes internas: [confirmación congelada](frozen_confirmation.json),
[lesión recurrente](recurrent_lesion.json), [matriz espacial completa](gpu_spatial_crossed/summary.json),
[routing lineal](routing_rank_audit.json), [rectificación](sensory_rectification_audit.json)
y [mecanismo de cobertura](rectification_support_audit.json).
La [bitácora](../docs/BITACORA.md) conserva resultados negativos y decisiones.

Se reutilizaron checkpoints; la última campaña añadió solo seis cruces que faltaban.
No se excluyeron semillas o clips para favorecer al conectoma. Las nueve celdas no
son nueve réplicas independientes: comparten entrenamiento/routing por fila y grafo
por columna. La semilla de entrenamiento todavía combina inicialización, rutas y
lotes. Los intervalos de la confirmación anterior no se trasladan a esta comparación.

## Límites que permanecen

Una columna de un animal, 1357 neuronas y una máscara entrenable por pareja neuronal;
no hay correspondencia funcional de células ni retinotopía medida. Los nulos conservan
intervalos de distancia entre somas, no distancias continuas, trayectorias axonales
ni mezcla certificada de la cadena de intercambios. La interfaz es aleatoria y el
núcleo contractivo. La validación de desarrollo ya se ha usado para tomar decisiones.
La partición con semilla 20261110 también está consumida para la confirmación histórica.

El diagnóstico de ReLU estudia un mapa local en un espacio de 32 características y
un solo fotograma. Sus colisiones construidas no se presentan como vídeos realizables.
No explica por sí solo la diferencia frente a GRU ni el resultado entre grafos.
Eliminar el defecto local podría alterar otras propiedades, incluida la no linealidad
que la tarea temporal necesita; no es automáticamente una mejora del modelo.

## Control futuro propuesto, no programado

Si una investigación posterior demuestra relevancia temporal del defecto sensorial,
un control posible sería **calibrar la inicialización de sesgo con entrenamiento**,
con las mismas rutas y el mismo procedimiento para el medido y todos sus nulos.
Una construcción concreta: para cada ruta y cada característica, medir en entrenamiento
la mayor preactivación de las neuronas que la reciben; elegir un único desplazamiento
no negativo de sesgo que cubra el peor caso, con un margen fijado antes de evaluar.
No usar etiquetas ni ajustar el desplazamiento con validación. Congelar y registrar
el resultado antes de cualquier comparación entre grafos.

Primero verificar en CPU cobertura, rango y fracción de neuronas activas; cubrir
columnas no garantiza por sí solo rango completo. Comprobar si la calibración vuelve
la dinámica casi lineal o reduce la información temporal utilizable. Es una modificación
de inicialización, no garantía de conservar rango tras entrenamiento: los sesgos
aprendidos pueden deshacerla. Una restricción permanente sería otro modelo y exigiría
su propio control. No entrenar únicamente el conectoma ni buscar un sesgo que invierta
el resultado anterior. Cualquier nueva confirmación requeriría otro plan y una partición
no consultada; el resultado negativo actual se conserva.

Este control se formula para hacer explícita una posibilidad metodológica. La rareza
del defecto y la falta de vínculo con el error temporal **no justifican ejecutarlo
ahora ni iniciar otra campaña GPU**. También podrían motivar una nueva fase datos
funcionales adecuados o una hipótesis concreta que necesite otra tarea, interfaz o
circuito. El presupuesto restante, por sí solo, no es motivo para continuar entrenando.

## Recursos y continuidad

47 tests pasan; código, máscaras, rutas, configuraciones y checkpoints relevantes
están auditados. Ocho campañas GPU completas y un intento rechazado, 66 ensayos GPU;
ninguna GPU BIONN activa al cerrar esta revisión. La auditoría actual y su seguimiento
usaron 16,02 s de CPU local y no entrenaron modelos.

Reservas conservadoras acumuladas: **18 USD de un límite operativo de 50 USD**,
con 32 USD de margen. No son la factura. La ampliación condicional a 75 USD no se
ha activado; las instantáneas de Modal siguen sujetas a conciliación. Ver el
[registro de gasto](compute_ledger.json).

El [estado de investigación](../docs/RESEARCH_STATE.md) determina la siguiente acción.
El cierre inicial no equivale a escalar. La nueva petición del usuario motivó el
seguimiento metodológico descrito a continuación; no revoca el resultado del piloto.
La automatización nocturna fue eliminada al finalizar la ventana a las 08:00
Europe/Paris del 10 de noviembre.

## Revisión solicitada por el usuario: continuar sin escalar

**Decisión comunicada:** piloto comparativo finalizado; siguiente paso, mejorar el
método a la escala actual. No aumentar circuito, complejidad de datos o GPU todavía.
El plan y sus condiciones están en [`docs/NEXT_PHASE.md`](../docs/NEXT_PHASE.md).

La GRU y los modelos de grafo usan las mismas 32 características visuales. La
GRU alcanza EPE 0,561851 con 18 946 parámetros y 500 pasos; el grafo, 1,088736 con
84 383 parámetros y 3000 pasos. No iguala arquitectura, entrada aprendida ni coste:
es una referencia de señal aprendible, no una atribución causal de la limitación.
El MSE del grafo pasa de 0,494262 en entrenamiento a 1,417213 en validación, frente
a 1,498047 de la predicción cero. La generalización merece atención antes del tamaño.

Un nuevo diagnóstico, fijado antes de ejecutar en
[`time_reversal_plan.json`](../configs/time_reversal_plan.json), reutilizó los 15
checkpoints (3 medidos, 9 nulos espaciales, 3 GRU), todos los 512 clips de validación
y los 128 contrafactuales. Ningún entrenamiento ni GPU. Se reprodujeron métricas
originales y se verificaron datos, calibración, grafos y rutas. Invertir estos vídeos
invierte exactamente la velocidad. La componente antisimétrica del predictor es
`[f(x)-f(reverse(x))]/2`.

| Modelo | EPE original | EPE proyección de dos pasadas | Reducción descriptiva |
| --- | ---: | ---: | ---: |
| Conectoma medido | 1.088736 | 0.976925 | 10.27 % |
| Nulos espaciales (3×3) | 1.090932 | 0.980304 | 10.14 % |
| GRU de referencia | 0.561851 | 0.486596 | 13.39 % |

El filtro prefijado de mejora ≥5 % se cumple en cada semilla medida (9,66–11,14 %)
y cada promedio por fila espacial (9,89–10,40 %), sin deterioro contrafactual medio.
El EPE contrafactual pasa de 1,070050 a 0,956579 en el medido y de 1,063102 a
0,962023 en los nulos. Las 15 identidades de datos y las métricas originales se
verificaron; las dos particiones se evaluaron en cada modelo. Resultados completos:
[`time_reversal_audit.json`](time_reversal_audit.json).

Esto identifica una componente del error incompatible con la simetría de la tarea,
no la causa de haberla aprendido. La proyección requiere dos evaluaciones del clip
completo. Es una mejora descriptiva de ese predictor retrospectivo en desarrollo;
no demuestra que se pueda obtener el beneficio con una pasada ni generalizar a
otra tarea. El MSE medio sobre una pareja y su inversión se descompone exactamente
en el MSE de la componente antisimétrica más la norma cuadrática de la componente
simétrica. Esa identidad algebraica no es una prueba estadística; la mejora en la
orientación original se midió por separado y no estaba garantizada.

El beneficio también aparece en los nulos y la GRU. Tras la proyección, el medido
supera descriptivamente a los nulos solo 0,34 %, con diferencias por fila
−0,001418 / −0,004118 / −0,004599. Son resultados de desarrollo posteriores,
no una confirmación anatómica nueva. Se mantiene la decisión de no escalar.

**Mejora elegida:** probar aumento por inversión temporal durante entrenamiento,
idéntico para medido y controles, conservando una sola pasada en evaluación.
La prueba está especificada pero no implementada ni lanzada. No añadir al mismo
tiempo otras modificaciones; no repetir campañas si fracasa sin evidencia nueva.
El defecto sensorial raro de ReLU no es la prioridad actual.

El seguimiento tardó 192,76 s locales (192,26 s CPU), dentro del límite de 300 s,
y pasan 49 tests. No añadió gasto ni reservas de Modal. Permanecen 18 USD reservados
de 50; no se ha activado el límite condicional de 75. Última consulta de facturación:
0,63059084 USD en aplicaciones efímeras de la cuenta, sin factura exclusiva definitiva.
