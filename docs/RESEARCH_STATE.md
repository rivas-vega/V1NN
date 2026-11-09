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
| La ventaja desaparece al conservar grados E/I por nodo. | Tres nulos válidos, 81,4–81,5 % de conexiones cambiadas. EPE de desarrollo 1,08874 medido vs 1,10281 E/I; diferencia −0,01407, signos consistentes. | No desaparece en este piloto; no es confirmación nueva. Auditar ahora grados conjuntos por tipo y distancia en ambos extremos. |
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

Entrada 012: v2 espacial aprobada en CPU con grados conjuntos exactos, 68,8–69,0 %
de aristas cambiadas y cobertura corregida en los dos estratos pequeños. No usar
los grafos v1, que se conservan como diagnóstico negativo de muestreo.
Preparada una campaña de solo tres ensayos espaciales a 3000 pasos; reserva nueva
2 USD, acumulado 16 USD y margen 34 USD. Consultar estado local y Modal para
comprobar si ya se lanzó; no duplicar. Al terminar, comparar con los controles
existentes usando `compare_type_nulls.py --control spatial_null`, verificar
identidad de código/checkpoints y aplicar el plan de la entrada 012.
La lesión CPU de los tres conectomas terminó: retirar corriente recurrente en
inferencia aumenta EPE medio +0,37102. No repetirla ni interpretarla como test de
especificidad anatómica o incapacidad del control sin aristas entrenado.

La partición de 2048 clips de la entrada 009 ya fue consultada. Conserva valor
como confirmación de los modelos congelados entonces, pero no puede reciclarse
como una confirmación independiente de futuros ajustes inspirados por sus resultados.
