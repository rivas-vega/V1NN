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
| La ventaja persiste al conservar organización E/I y espacial. | Diseño cruzado 3×3 completo: EPE medido 1,088736 vs nulo 1,090932; diferencias por fila −0,004261 / −0,009570 / +0,007242. Mejora media 0,20 %, MSE 0,24 % peor. | Comparación cerrada sin ventaja robusta del cableado fino. No repetir cruces ni buscar semillas/pasos favorables; no justifica confirmación nueva o escalado. |
| Los modelos aprendidos usan corriente recurrente. | Lesión CPU de tres checkpoints: EPE aumenta +0,37102 de media al retirar únicamente corriente recurrente en inferencia. | Dependencia demostrada dentro de esos predictores; no es superioridad anatómica ni comparación de capacidad del modelo sin aristas. No repetir la misma lesión. |
| Eliminar la fuga (`alpha=1`) corrige el problema. | Empeora conectoma, sin aristas y nulo en tres semillas. | Descartado en el régimen probado. Reabrir solo por una interacción nueva y explícita, no repetir la misma campaña. |
| El conectoma puede memorizar 32 ejemplos. | Campaña 006 completada: conectoma y sin aristas cumplen <1 % del error cero en las tres semillas; validación ≈2,98. | Superada la capacidad de memorizar 32 clips. No repetirla ni confundirla con generalización. |
| Resolver la lectura lineal recupera una representación temporal útil. | 30 lecturas sobre núcleos congelados: mejor regularización se aproxima a la predicción constante; mínimos cuadrados sobreajusta. | No rescata estos estados. No repetir el barrido sin cambiar o justificar la representación. |
| El fallo a 2048 ejemplos depende del tiempo de optimización conjunta. | Ajuste pequeño posible; lectura congelada insuficiente; 500 pasos dejan entrenamiento alto. | Campañas 008/009 completas: 3000 pasos permiten aprender y la mejora se mantiene en clips nuevos. No aumentar pasos de nuevo por inercia; continuar con controles de estructura. |
| El routing aleatorio elimina dimensiones antes de la rectificación. | Rango 32/32 en las tres rutas; reconstrucción de retina con error relativo <1,3e-8, también con ganancias aprendidas. | Descartado en esta etapa lineal. La auditoría posterior encuentra pérdida ocasional después de ReLU, no un fallo general de rango. |
| La primera rectificación pierde dimensiones sensoriales. | Auditoría CPU: rango deficiente en 0,78–1,95 % de primeras imágenes de validación por ruta en los modelos aprendidos; todas las pérdidas se explican por características sin rutas activas. Diez colisiones exactas verificadas. | Mecanismo local documentado; no demuestra pérdida de movimiento ni causa del resultado entre grafos. No justifica nueva GPU ni excluir clips. Propuesta futura de calibración no programada. |

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

Entrada 016 completa. Piloto comparativo y diagnósticos inmediatos cerrados:
matriz espacial cruzada sin ventaja robusta, routing lineal de rango completo,
pérdida ocasional tras primera ReLU explicada por columnas sin rutas activas.
No repetir matriz, nulos, lesión, auditoría de routing, SVD ni soporte sensorial.
No queda análisis local o campaña en curso. Reservas 18 USD, margen 32 bajo 50;
ninguna GPU BIONN activa y ampliación a 75 sin usar.

No hay un experimento inmediato científicamente justificado. Leer
`reports/PHASE1_REVIEW.md` para síntesis, límites y condiciones de reapertura.
Su control de calibración es una propuesta no programada; no ejecutarlo por
figurar allí. El campo `planned_not_run` de `configs/sensory_rectification_plan.json`
es parte del plan histórico congelado: **ese análisis ya se ejecutó**, con resultados
en `reports/sensory_rectification_audit.json` y entrada 016.

En siguientes revisiones nocturnas, comprobar si existe nueva instrucción, evidencia
o ejecución activa. Si no cambió nada accionable, mantenerse en silencio sin
repetir pruebas o gastos. Al alcanzar el 10 de noviembre a las 08:00 Europe/Paris,
desactivar la automatización. No interpretar una tarea de vigilancia como necesidad
de inventar otro experimento, consumir presupuesto o reiniciar vías descartadas.

La partición de 2048 clips de la entrada 009 ya fue consultada. Conserva valor
como confirmación de los modelos congelados entonces, pero no puede reciclarse
como una confirmación independiente de futuros ajustes inspirados por sus resultados.
