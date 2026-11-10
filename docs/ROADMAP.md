# Hoja de ruta y decisiones

## Implementación frente a investigación

| Área | Disponible | Pendiente |
| --- | --- | --- |
| Datos | Importación real con hashes verificados, agregación e IDs exactos, fixtures | Auditoría funcional, fronteras, matches funcionales |
| Modelo | RNN dispersa, E/I, entrada fija, contracción, lectura lineal | Routing fisiológico, microsteps, priors por tipo |
| Objetivos | Movimiento 2D, SSL sensorial, pérdidas de observación | Flujo denso, observación calibrada y sincronizada |
| Controles | GRU/CNN, historia lineal, no-edge/reset, nulos de grados, E/I y espacio; lesión recurrente CPU | Perturbaciones y confirmación anatómica completa; reservorio disponible en código, no equipararlo a un control ya evaluado |
| Validación | 47 tests, identidades y checkpoints auditados; comparaciones de tres semillas | Ventaja robusta del cableado fino, fisiología y conciliación de factura definitiva |

## P0 — Contrato de datos

Descarga real y recuentos verificados el 2026-11-09; detalle en la [bitácora](BITACORA.md).

Ejecutar el importador, verificar identidades, signos desconocidos, autapsis, aislamiento, unidades y licencia. Salida: manifiesto real auditado. Escribir un script no completa esta fase.

## P1 — Benchmark temporal válido

Piloto implementado y comparado en tres semillas CPU/GPU; GRU y CNN temporal aprenden. Queda la evaluación de generalización fuera del régimen de entrenamiento. Detalle y límites en la [bitácora](BITACORA.md).

Implementar historia lineal y una GRU/CNN que aprendan. Verificar que la imagen estática no revela la etiqueta. Separar semillas y escenas; añadir historias distintas con igual imagen final. La prueba smoke no sustituye a este benchmark.

## P2 — Comparación de grafos

Tras ampliar de 500 a 3000 pasos, se observó una ventaja pequeña frente al nulo de grados, confirmada en 2048 clips nuevos. El control E/I y espacial posterior reduce la diferencia y la vuelve inconsistente entre semillas. Se ejecutó un diseño fijo para separar realización nula del paquete de entrenamiento/routing; consultar su resultado más reciente en [el estado](RESEARCH_STATE.md). La confirmación anterior no incluye los controles E/I y espaciales. No escalar por el resultado de un control más débil ni volver a probar alpha=1 sin evidencia nueva.

Auditar nulos y cambio efectivo bajo invariantes. Usar misma entrada, lectura, dinámica y presupuesto. Movimiento primero; repetir luego con SSL sensorial como estudio independiente. Registrar curvas, semillas, escenas y coste. No mezclar pérdidas retrospectivamente para rescatar un resultado.

## P3 — Fisiología

Identificar matches reales, sincronizar estímulos, calibrar observación y evaluar en vídeos reservados. Separar lectura por célula de lectura flexible. Un modelo fundacional neural es referencia externa, no teacher oculto.

## P4 — Generalización y mecanismos

Oclusiones, contraste, velocidades/texturas nuevas; lesiones y reentrenamiento; sensibilidad a routing, núcleo/halo y estabilidad. Escalar solo por una pregunta que requiera más circuito o evidencia reproducible.

## Decisiones abiertas

Cuántas células emparejadas quedan en la columna; qué routing puede calibrarse; si los nulos mezclan suficientemente; pesos por arista frente a compartidos; si SSL aporta más que persistencia; coste real en hardware disponible. No presentar estas incógnitas como decisiones ya resueltas.

## Reproducibilidad

Guardar source commit, grafo y hashes, splits, configuraciones, semillas, selección de checkpoints y presupuesto de búsqueda. El plan 5×5×3 no es un experimento ejecutado ni una prerregistración. Fijar análisis antes de usar el test final.

Mantener BIONN independiente de GDA y conservar en archivo las ramas humana sintética, mosca y liquid networks. La revisión metodológica con simuladores corticales recientes y reglas locales es necesaria antes de reclamar novedad.
