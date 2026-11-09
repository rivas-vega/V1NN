# Protocolo científico de V1NN

## Pregunta y alcance

¿La conectividad dirigida medida en una columna de V1 facilita una computación visual temporal frente a grafos que conservan sus propiedades estructurales principales, cuando entrada, dinámica neuronal y lectura están controladas?

El objetivo es identificar un efecto atribuible al grafo **dentro del modelo**, no demostrar que el córtex sea óptimo, reconstruir la experiencia del animal ni desarrollar un world model general. BIONN conserva sus ramas históricas; V1NN aborda esta pregunta concreta e independiente de GDA.

## Tres estudios separados

**A. Capacidad computacional, principal.** Entrenar pesos recurrentes únicamente en aristas observadas para estimar movimiento. Etiquetas del simulador, filtros de entrada fijos y lectura lineal. La supervisión hace explícita la computación exigida y evita transferirla desde un teacher.

**B. Objetivo autosupervisado, secundario.** Conservar arquitectura y condiciones comparables, pero predecir una transformación sensorial fija de un fotograma futuro. Evaluar movimiento mediante una lectura con el núcleo congelado. Disminuir el error sensorial no implica aprender una representación útil.

**C. Correspondencia biológica.** Congelar redes entrenadas en A/B, presentar estímulos experimentales y contrastar células identificadas en ambas modalidades. Ajustar únicamente una observación pequeña sobre una partición de calibración. Ajustar todos los pesos a registros neuronales es otro estudio de identificación supervisada de sistemas, no correspondencia emergente.

## Interfaz sin atajos

El flujo es `fotograma -> filtro espacial fijo -> distribución de entrada -> estado recurrente -> lectura lineal`. La lectura no accede directamente a píxeles, características visuales externas, fotogramas futuros o ventanas temporales concatenadas. Los filtros no tienen memoria ni normalización entre instantes. Se reutilizan exactamente la misma transformación y rutas de entrada al comparar grafos.

Esto elimina ciertos atajos, pero **no demuestra que se utilicen las conexiones**. La fuga de una neurona conserva memoria incluso sin aristas. Comparar controles entrenados desde cero y lesiones en inferencia: las dos intervenciones responden a preguntas diferentes.

El routing aleatorio fijo implementado es diagnóstico, no retinotopía. Una columna estrecha no representa automáticamente toda una pantalla. No mapear posiciones de somas a píxeles sin evidencia. Para conclusiones sobre V1 hay que calibrar campos receptivos con datos de entrenamiento o estudiar explícitamente la sensibilidad a distintas interfaces. Aprender una proyección completamente libre puede absorber el efecto del grafo.

## Tarea inicial y progresión

El generador desplaza texturas aleatorias periódicamente en ocho direcciones. Cada imagen aislada tiene la misma distribución marginal con independencia de la velocidad. Así se elimina un atajo estático evidente. La implementación estima **velocidad global 2D**, no flujo óptico denso.

Antes de evaluar MICrONS, comprobar que un modelo temporal de referencia aprende y que una lectura estática no explota artefactos. Añadir historias distintas con idéntica imagen final. Separar texturas, velocidades y semillas entre entrenamiento, validación y test. El piloto actual separa texturas mediante semillas de generación y añade historias con idéntico fotograma final. Comparte las ocho velocidades entre entrenamiento y validación: la extrapolación a velocidades nuevas y el test final independiente siguen pendientes.

Después introducir parches con movimiento independiente, oclusiones, pérdidas de fotogramas y cambios de contraste. Distinguir movimiento visible de mantenimiento de una estimación durante ausencia de entrada. La segunda tarea exige memoria, no percepción de una imagen observable. El flujo denso y escenas naturales son extensiones, no capacidades ya implementadas.

## Comparaciones necesarias

| Comparación | Qué distingue |
| --- | --- |
| Medido vs aleatorio con N y E iguales | Efecto más allá de tamaño y sparsity |
| Medido vs grados dirigidos preservados | Efecto más allá de hubs y distribución de grados |
| Medido vs grados, grupos y distancias preservados | Efecto más allá de organización espacial y celular gruesa |
| Sin aristas, manteniendo fuga | Memoria individual frente a interacción entre neuronas |
| Estado reiniciado en cada fotograma | Necesidad de historia temporal |
| Lectura estática e historia lineal | Atajos visuales y memoria lineal suficiente |
| CNN temporal y GRU | Alternativas prácticas competitivas |
| Grafo sin entrenar con lectura entrenada | Efecto de reservorio frente a plasticidad recurrente |

Solo llamar a un nulo «preservador de capas» si esa información existe y se utiliza realmente. El código permite grupos explícitos, no inventa capas cuando faltan.

Los intercambios dirigidos `(a,b),(c,d) -> (a,d),(c,b)` preservan grados por nodo y opcionalmente grados por grupo y bins de distancia. Registrar aceptación, intentos, fracción cambiada y sensibilidad al número de intercambios. La implementación no garantiza muestreo uniforme ni mezcla suficiente. Un nulo casi idéntico al original no proporciona un test fuerte; uno que destruye toda ruta entrada-salida tampoco es justo.

Mantener posición, identidad, signos e interfaz al cambiar el grafo. Entrenar cada nulo desde cero con igual presupuesto. La lesión de una red ya entrenada muestra dependencia de ese modelo, no incapacidad de aprendizaje de la arquitectura lesionada.

## Parámetros y grados de libertad

El prototipo aprende un peso por pareja neuronal. Conserva recuentos de sinapsis como metadatos, pero no los usa como fuerza fisiológica. Es un estudio de máscara. Comparar después pesos compartidos por tipo y priors basados en recuento: demasiados pesos libres pueden compensar cambios de conectividad.

No siempre pueden igualarse simultáneamente estado, parámetros, operaciones y tiempo. Reportar los cuatro y dar el mismo presupuesto de selección de hiperparámetros. Una búsqueda exhaustiva para MICrONS y superficial para controles no mide una ventaja del cableado.

## Métricas y estadística

Para A, proponer error euclídeo de velocidad en clips nuevos como criterio principal, con MSE de entrenamiento, error angular solo en velocidades no nulas y curvas de eficiencia de datos. Para B, error relativo a persistencia y media de entrenamiento, seguido de sonda lineal de movimiento en datos distintos. Para C, correlación por célula, error calibrado y fracción de varianza explicable solo cuando existen repeticiones y una estimación fiable de ruido.

Reportar rendimiento limpio y degradación bajo perturbación. Una caída porcentual menor no es robustez si se parte de una precisión inferior. Emparejar escenas e inicializaciones compatibles entre grafos.

Separar variación por entrenamiento, realización del grafo nulo y routing. El plan inicial 5 semillas × 5 nulos × 3 rutas es orientativo, no un cálculo de potencia ni un estudio realizado. Comenzar con un piloto y congelar presupuesto y análisis antes del test final.

Los fotogramas no son observaciones independientes. Los intervalos deben respetar la jerarquía escenas, vídeos y sesiones. Varias semillas no equivalen a varios animales. Una columna de un animal no demuestra universalidad cortical. Guardar manifiestos y hashes de particiones.

## Criterios para continuar

1. Ingeniería: gradientes correctos, identidad verificable y un baseline temporal que resuelve la tarea.
2. Estudio anatómico: grafo real auditado, interfaz explícita y nulos efectivos.
3. SSL: conservarlo si aporta información o transferencia más allá de media y persistencia; no conservarlo por preferencia estética.
4. Escalado: ampliar solo ante una señal consistente o una pregunta que requiera más circuito.

Superar al azar simple pero no al nulo espacial/tipado sugiere que basta la estructura gruesa. No superar controles limita la hipótesis en el régimen probado; no demuestra que los conectomas carezcan de valor. Una pérdida o entrada que destruyen información relevante pueden invalidar el test antes de comparar grafos.

## Preguntas adversariales

¿La etiqueta se infiere de un solo fotograma? ¿La lectura oculta memoria? ¿La red utiliza aristas o solo fuga? ¿El nulo conserva la interfaz? ¿Los pesos compensan cualquier cableado? ¿Las fronteras explican el efecto? ¿La ventaja cambia con filtros o routing? ¿Hay filtración entre vídeos? ¿Los controles tuvieron igual oportunidad de ajuste? ¿Se afirma más de lo que permite un animal?

## Antecedentes y novedad

[Lappalainen et al. 2024](https://www.nature.com/articles/s41586-024-07939-3) ya relacionan conectividad, movimiento y respuestas neuronales. [Ding et al. 2025](https://www.nature.com/articles/s41586-025-08840-3) estudian reglas de conectividad funcional. [Wang et al. 2025](https://www.nature.com/articles/s41586-025-08829-y) predicen respuestas corticales mediante un modelo fundacional. El [preprint cortical de 2026](https://pubmed.ncbi.nlm.nih.gov/42079060/) es un antecedente cercano que requiere comparación metodológica detallada; no presentarlo como validación de esta hipótesis.

La contribución potencial es evaluar **especificidad del cableado bajo distintos objetivos y controles de interfaz**, no construir «la primera RNN biológica». La prioridad bibliográfica no está demostrada.

## Límite específico del control sin aristas

En esta implementación, sin aristas el estado cumple
`h_t = (1-alpha) h_(t-1) + alpha * ReLU(sensory_t + bias)`.
Por tanto la salida final es una suma ponderada de funciones de cada fotograma
por separado, más el sesgo de lectura. Conserva memoria por fuga, pero no genera
interacciones no lineales entre fotogramas. La identidad de aditividad al combinar
mitades de historias está comprobada en los tests.

Como las texturas del generador son estacionarias y la distribución marginal de
cada fotograma no depende de la velocidad, cada término de esa suma tiene la
misma esperanza condicionada a cada velocidad. En la población equilibrada no
puede reducir el MSE de la predicción constante mediante correlaciones temporales.
Puede memorizar correlaciones accidentales de una muestra finita, como mostró
la campaña de 32 clips. Esto explica por qué ajustar entrenamiento en este
control no demuestra aprendizaje del movimiento. El conectoma sí puede producir
interacciones entre instantes mediante la recurrencia dentro de la no linealidad;
que esas interacciones sean útiles debe comprobarse experimentalmente.
