# Estado de investigación y criterios de continuidad

Este documento resume decisiones; la evidencia completa y su historia permanecen
en `BITACORA.md` y `../reports/`. Actualizar ambos después de cada campaña. Una
revisión programada comienza por el estado local de continuación y por comprobar
si hay trabajo en curso; no vuelve a empezar automáticamente el experimento de
una entrada antigua.

| Hipótesis o pregunta | Evidencia actual | Decisión y condición para reabrir |
| --- | --- | --- |
| La señal sensorial permite aprender movimiento temporal. | GRU y CNN temporal mejoran claramente la referencia cero en tres semillas; lectura estática no lo hace. | Superada en este piloto. Revalidar si cambia el generador o la interfaz. |
| MICrONS supera al nulo de grados con la interfaz inicial. | A 500 pasos, alpha 0,3 y 2048 clips, ambos quedan por encima del error cero; sus errores son similares. | Sin evidencia de ventaja. No ampliar semillas/nulos antes de establecer un régimen que aprenda. |
| Eliminar la fuga (`alpha=1`) corrige el problema. | Empeora conectoma, sin aristas y nulo en tres semillas. | Descartado en el régimen probado. Reabrir solo por una interacción nueva y explícita, no repetir la misma campaña. |
| El conectoma puede memorizar 32 ejemplos. | Campaña 006 completada: conectoma y sin aristas cumplen <1 % del error cero en las tres semillas; validación ≈2,98. | Superada la capacidad de memorizar 32 clips. No repetirla ni confundirla con generalización. |
| Resolver la lectura lineal recupera una representación temporal útil. | 30 lecturas sobre núcleos congelados: mejor regularización se aproxima a la predicción constante; mínimos cuadrados sobreajusta. | No rescata estos estados. No repetir el barrido sin cambiar o justificar la representación. |
| El fallo a 2048 ejemplos depende del tiempo de optimización conjunta. | Ajuste pequeño posible; lectura congelada insuficiente; 500 pasos dejan entrenamiento alto. | Próximo diagnóstico: 3000 pasos, sin cambiar otros factores, según entrada 007. Aún no ejecutado ni reservado. |

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

## Próxima acción concreta

Leer el plan y las ramas de decisión al final de la entrada 007. Comprobar primero
si ya se ejecutó desde otra revisión. Solo después de comprobar Modal y reservar
2 USD, lanzar una campaña de nueve ensayos con los límites indicados. Tras ella,
actualizar esta tabla; no aumentar pasos de nuevo por inercia.
