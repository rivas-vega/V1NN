# Estado de investigación y criterios de continuidad

Este documento resume decisiones; la evidencia completa y su historia permanecen
en `BITACORA.md` y `../reports/`. Actualizar ambos después de cada campaña. Una
revisión programada comienza por el estado local de continuación y por comprobar
si hay trabajo en curso; no vuelve a empezar automáticamente el experimento de
una entrada antigua.

| Hipótesis o pregunta | Evidencia actual | Decisión y condición para reabrir |
| --- | --- | --- |
| La señal sensorial permite aprender movimiento temporal. | GRU y CNN temporal mejoran claramente la referencia cero en tres semillas; lectura estática no lo hace. | Superada en este piloto. Revalidar si cambia el generador o la interfaz. |
| MICrONS supera al nulo de grados con la interfaz inicial. | A 500 pasos no se veía ventaja; a 3000, confirmación en 2048 clips nuevos: EPE 1,07241 vs 1,09245, diferencia −0,02004 con intervalo por clips [−0,02526, −0,01508]. | Ventaja pequeña y condicionada. El control E/I posterior tiene solo evaluación de desarrollo; no atribuir aún el efecto al cableado fino. |
| La ventaja desaparece al conservar grados E/I por nodo. | Tres nulos válidos, 81,4–81,5 % de conexiones cambiadas. EPE de desarrollo 1,08874 medido vs 1,10281 E/I; diferencia −0,01407, signos consistentes. | No desaparece con E/I solo, pero el control espacial posterior reduce la diferencia y cambia su signo por semilla. No es confirmación nueva. |
| La ventaja persiste al conservar organización E/I y espacial. | Nulos v2 auditados: EPE medido 1,08874 vs espacial 1,09264; diferencias −0,00996 / −0,01055 / +0,00881. MSE medio ligeramente peor para medido. | No hay ventaja consistente. Completar diseño fijo cruzado para separar realización nula del paquete de semilla de entrenamiento/routing; no retocar hiperparámetros. |
| Los modelos aprendidos usan corriente recurrente. | Lesión CPU de tres checkpoints: EPE aumenta +0,37102 de media al retirar únicamente corriente recurrente en inferencia. | Dependencia demostrada dentro de esos predictores; no es superioridad anatómica ni comparación de capacidad del modelo sin aristas. No repetir la misma lesión. |
| Eliminar la fuga (`alpha=1`) corrige el problema. | Empeora conectoma, sin aristas y nulo en tres semillas. | Descartado en el régimen probado. Reabrir solo por una interacción nueva y explícita, no repetir la misma campaña. |
| El conectoma puede memorizar 32 ejemplos. | Campaña 006 completada: conectoma y sin aristas cumplen <1 % del error cero en las tres semillas; validación ≈2,98. | Superada la capacidad de memorizar 32 clips. No repetirla ni confundirla con generalización. |
| Resolver la lectura lineal recupera una representación temporal útil. | 30 lecturas sobre núcleos congelados: mejor regularización se aproxima a la predicción constante; mínimos cuadrados sobreajusta. | No rescata estos estados. No repetir el barrido sin cambiar o justificar la representación. |
| El fallo a 2048 ejemplos depende del tiempo de optimización conjunta. | Ajuste pequeño posible; lectura congelada insuficiente; 500 pasos dejan entrenamiento alto. | Campañas 008/009 completas: 3000 pasos permiten aprender y la mejora se mantiene en clips nuevos. No aumentar pasos de nuevo por inercia; continuar con controles de estructura. |

## Reglas para evitar bucles

1. Antes de ejecutar, escribir qué incertidumbre se resolverá y qué resultados
   cambiarían la siguiente decisión. Si ningún resultado cambiaría nada, no lanzar.
2. Reutilizar checkpoints y hacer análisis en CPU cuando no haga falta entrenar.
3. Cambiar un factor de forma controlada, emparejar datos y semillas, y distinguir
   el ajuste del método de una evaluación final. Conservar toda búsqueda negativa.
4. Volver a una vía anterior solo con evidencia nueva, error de implementación o
   imposibilidad documentada de la vía elegida; indicar el motivo en la bitácora.
5. No gastar para llenar la noche. Si una incertidumbre requiere información que
   falta, registrar el bloqueo y avanzar únicamente en tareas independientes útiles.
6. Mantener el límite acumulado de 50 USD, reservas previas, ejecución secuencial
   acotada y consulta de aplicaciones activas. Los créditos no amplían autorización.
   Existe autorización condicional hasta 75 USD solo por necesidad estricta,
   con conciliación y justificación previa. Aún no se ha activado.

## Próxima acción concreta

Entrada 014: semilla nula independiente implementada y auditoría CPU completada.
La diagonal reproduce exactamente código histórico, métricas, pesos y optimizador
en tres integraciones reducidas; los seis cruces conservan routing e inicialización
por semilla de entrenamiento y seleccionan el grafo correcto. Plan congelado en
`configs/spatial_crossed_plan.json`; su campo de reserva describe el momento de
prerregistro, el estado vigente de gasto está en `reports/compute_ledger.json`.

Se han reservado 2 USD para seis cruces nuevos: 18 acumulados, margen 32 bajo 50.
No ampliar a 75. Ninguna GPU BIONN estaba activa al verificar consumo; publicar
código verificado y lanzar una sola campaña de 1200 s, luego analizar la matriz
completa y actualizar este estado. No repetir diagonal ni generar más nulos.
Si el efecto sigue siendo inconsistente, cerrar esta comparación y auditar la
interfaz sensorial antes de escalar; no buscar semillas o hiperparámetros favorables.

La partición de 2048 clips de la entrada 009 ya fue consultada. Conserva valor
como confirmación de los modelos congelados entonces, pero no puede reciclarse
como una confirmación independiente de futuros ajustes inspirados por sus resultados.
