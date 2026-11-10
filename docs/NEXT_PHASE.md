# Continuación decidida después del piloto

Decisión del 10 de noviembre de 2026, entrada 017. **El piloto comparativo inicial
está cerrado. No comienza el escalado.** La siguiente fase es una mejora metodológica
acotada, con el mismo circuito y la misma tarea. El usuario debe recibir un aviso
expreso antes de cualquier transición futura al escalado.

## Por qué esta prioridad

El conectoma medido y el nulo espacial completo tienen errores prácticamente iguales
en desarrollo: EPE 1,088736 y 1,090932. Una GRU con la misma retina fija alcanza
0,561851. No es una comparación que iguale arquitectura, parámetros u operaciones:
la GRU tiene 18 946 parámetros y 500 pasos; el modelo de grafo, 84 383 y 3000.
La GRU aprende su proyección de entrada y usa otra dinámica. Sí demuestra que las
32 características de la retina contienen señal temporal utilizable. No atribuir
toda la brecha a la retina, a la ReLU, al conectoma o a falta de pasos.

El grafo presenta MSE de entrenamiento 0,494262 y de validación 1,417213; la
predicción cero tiene MSE 1,498047. Hay una brecha de generalización importante,
pero ese patrón por sí solo no identifica su causa ni garantiza que una técnica
de regularización la resuelva. Cambiar escala ahora mantendría esa incertidumbre.

El diagnóstico CPU posterior aporta un candidato más concreto. Invertir el tiempo
de una textura en traslación cambia el signo de su velocidad. La proyección
`[f(x)-f(reverse(x))]/2` reduce el EPE en 10,27 % en el medido y 10,14 % en los nulos;
la GRU también mejora 13,39 %. Se cumple el filtro práctico fijado antes del análisis
en las tres semillas y los controles contrafactuales. Ver
[`time_reversal_audit.json`](../reports/time_reversal_audit.json).

Es una intervención sobre el predictor congelado, con dos pasadas retrospectivas.
No demuestra que un entrenamiento con ejemplos invertidos mejore una sola pasada,
ni que el beneficio sea propio del cableado medido. La ventaja descriptiva entre
grafos después de proyectar es solo 0,34 %; aunque sus tres diferencias por fila
son negativas, procede de la misma validación consultada y no reabre una afirmación
de ventaja robusta. El piloto negativo se conserva.

## Próximo experimento: un único cambio

**Hipótesis:** añadir inversión temporal como aumento de datos durante entrenamiento
reduce el error de una predicción causal de una sola pasada en ambos tipos de grafo.
En cada clip seleccionado, probabilidad fija 0,5 de invertir todos los fotogramas
y negar ambas componentes de velocidad. Sin nuevos filtros, sesgos, capas, etiquetas
externas, tamaño de circuito ni búsqueda de tasas de aprendizaje.

1. Implementar una opción desactivada por defecto y verificar en CPU que el modo
   antiguo conserva exactamente rutas, inicialización y secuencia de lotes. El sorteo
   del aumento tendrá un generador separado, con semillas fijadas 107/108/109 para
   los ensayos 7/8/9. Probar transformación de etiquetas, reversibilidad y ausencia
   de aumento o recalibración en validación. No cambiar simultáneamente la interfaz.
2. Congelar código y un manifiesto ejecutable antes de entrenar. Proponer nueve
   ensayos nuevos: tres grafos medidos, tres nulos espaciales con parejas diagonales
   (7,7), (8,8), (9,9), y tres GRU. Reutilizar sus nueve resultados originales como
   controles sin aumento solo tras verificar identidad de datos, normalización,
   máscaras, inicialización, lotes y optimizador. Si esa identidad falla, detener
   el lanzamiento y corregirla; no mezclar versiones silenciosamente.
3. Mantener 3000 pasos en grafos y 500 en GRU, lotes de 64, tasa 0,003, `alpha=0,3`
   donde corresponda, semillas y particiones originales. Es un estudio del efecto
   del aumento dentro de cada arquitectura, no una comparación de eficiencia igualada.
   Las tres parejas diagonales son un filtro metodológico barato; no sustituyen al
   diseño cruzado completo para atribuir un efecto a la anatomía.
4. Evaluar EPE de una sola pasada como criterio principal, además de MSE, error de
   entrenamiento, discrepancia al invertir el tiempo y control contrafactual. Pasar
   a validación prospectiva del método solo si reduce EPE al menos 5 % frente a su
   control original en las tres semillas medidas y las tres espaciales, sin empeorar
   el EPE contrafactual medio de ninguna familia. Publicar todos los resultados.
   El umbral es una decisión práctica de desarrollo, no un cálculo de potencia.
5. Si falla, cerrar esta variante; no buscar otras probabilidades, semillas o pasos
   para rescatarla sin una nueva hipótesis documentada. Si pasa, fijar un protocolo
   prospectivo que separe semilla de rutas, inicialización/lotes y grafo nulo antes
   de gastar en una comparación anatómica nueva. No consumir todavía otra partición
   de confirmación. La semilla histórica 20261110 ya fue consultada.

**Estado de este experimento:** propuesto y elegido como siguiente prueba; todavía
no implementado, reservado ni lanzado. El análisis CPU sí está completo y no se repite.
La petición actual de análisis queda resuelta con este plan; no se activa una nueva
campaña automática por haber terminado el diagnóstico.

## Coste y condiciones de escalado

Antes de entrenar: comprobar facturación y aplicaciones activas, conciliar reservas,
verificar margen y reservar 2 USD conservadores para una sola campaña T4 secuencial.
Prever 1200 s de trabajo, corte del servidor 1800 s, máximo 300 s por ensayo, un
contenedor y cero reintentos automáticos. Si no cabe, conservar resultados parciales
y decidir de nuevo; no ampliar límites automáticamente. Esta reserva futura **no
se ha sumado** a las actuales. Siguen siendo 18 de 50 USD; la extensión a 75 no se usa.

Superar el filtro de aumento de datos no autoriza científicamente el escalado.
Para estudiar más tejido, escenas naturales u objetivos autosupervisados debe existir
una pregunta que exija esa ampliación, o una señal reproducible frente al nulo espacial
fuerte con interfaz y selección comparables. Antes del test se fijarán efecto mínimo
relevante, tamaño de muestra y análisis de incertidumbre que respete rutas, semillas,
grafos y clips; los clips de una misma pareja no son réplicas independientes. Una
columna de un animal seguirá sin sostener generalizaciones biológicas.

Prioridades posteriores condicionadas: separar fuentes de variación; comprobar
transferencia a texturas y velocidades no consultadas; y obtener campos receptivos
medidos si se busca correspondencia biológica. La calibración de sesgo para pérdidas
ReLU raras sigue sin programarse porque no se ha vinculado al error de movimiento.
No aumentar neuronas, resolución, duración, campañas o presupuesto por inercia.

La revisión nocturna terminó y fue eliminada a las 08:00 Europe/Paris del 10 de
noviembre. Este plan no crea otra automatización. Avisar al usuario si posteriormente
se decide escalar, indicando evidencia, alcance y presupuesto antes de comenzar.
