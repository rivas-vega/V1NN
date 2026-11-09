# Pérdidas y aprendizaje bioinspirado

## Qué queremos aprender

Una pérdida debe corresponder a una pregunta. No mezclar cinco objetivos porque suenen biológicos. Se empieza con una computación visual medible, se añade un objetivo sin etiquetas como comparación y se reserva la fisiología para un análisis independiente.

## A. Movimiento supervisado

Con estado recurrente h y lectura lineal de velocidad v_hat = Ch + b:

$$L_{motion}=|O|^{-1}\sum_{(b,t)\in O}\|\hat v_{bt}-v_{bt}\|_2^2.$$

Las etiquetas son velocidades conocidas del generador. No hay anotación humana, pero el aprendizaje es **supervisado**. En imágenes sintéticas exactas, MSE es transparente. Para ruido u outliers, Huber sería una sensibilidad predefinida. La máscara no debe seleccionarse según qué predicciones resulten correctas. Evaluar error euclídeo y angular aparte de la función entrenada.

El precedente directo es [Connectome-constrained networks predict neural activity across the fly visual system](https://www.nature.com/articles/s41586-024-07939-3). Motiva la estrategia, no garantiza resultados en mamíferos. La implementación inicial no estima flujo denso ni comprueba física de objetos.

## B. Predicción sensorial sin teacher

Definir una transformación q fija y sin memoria: contraste centro-periferia con canales ON/OFF. El alumno solo recibe q(x_1),...,q(x_t):

$$L_{sensory}=|O_h|^{-1}\sum_{(b,t)\in O_h}\rho_\delta(C_hh_{bt}-q(x_{b,t+h})).$$

El código usa smooth L1 con beta 0.1 y un horizonte. Los horizontes físicos requieren timestamps correctos; el runner multihorizonte queda pendiente. El objetivo no se mueve conjuntamente con el alumno, por lo que no colapsa mediante adaptación del teacher. **El alumno sí puede producir una constante competitiva** si la información futura es débil o ambigua.

Comparar obligatoriamente contra la última señal y la media calculada solo en entrenamiento. Evaluar varianza y residuos, y realizar una sonda de movimiento sobre el núcleo congelado. No añadir VICReg automáticamente para generar variación que no mejora predicción.

Una extensión es predecir el residual q(x_(t+h))-q(x_t): la predicción cero se convierte en persistencia y obliga a examinar innovaciones. No elimina futuros ambiguos. La normalización se fija en entrenamiento y nunca utiliza estadísticas de todo el clip. Parches ocultos y predicción futura son protocolos diferentes.

Con vídeo natural, reconstruir píxeles puede penalizar textura impredecible y promediar futuros. Una pérdida fotométrica mediante warping permitiría aprendizaje de movimiento sin etiquetas, pero añade constancia de brillo, visibilidad y suavidad. Esos supuestos pueden explicar el resultado; no es la prueba inicial más limpia del grafo.

## Por qué no comenzar por JEPA

Un teacher potente aporta conocimiento visual adicional. Congelarlo iguala objetivos, pero no demuestra necesidad del conectoma: el encoder puede transformar la tarea en una lectura estática sencilla. JEPA queda como benchmark posterior de transferencia, no explicación del funcionamiento de V1NN. La hipótesis anatómica debe sostenerse primero con una interfaz limitada y controles de memoria.

## Revisión de SSL bioinspirado

| Método | Idea aprovechable | Límite para este proyecto |
| --- | --- | --- |
| Kermani Nejad et al. 2025 | Predicción contextual repartida entre capas | Modelo laminar específico, no pérdida universal para MICrONS |
| PredNet | Regularidades temporales sin etiquetas | Arquitectura jerárquica de errores diferente; reconstrucción no identifica cableado |
| CPC | Distinguir características futuras frente a negativos | Puede favorecer identidad de vídeo y otros atajos |
| Slow Feature Analysis | Factores temporales estables | La lentitud puede eliminar dirección o velocidad |
| Predictive coding | Asignación de crédito mediante errores locales | Requiere dinámica y realimentación explícitas |
| Equilibrium propagation | Relacionar objetivos y dinámica de equilibrio | No se aplica a una RNN arbitraria cambiando solo la pérdida |

[Self-supervised predictive learning accounts for cortical layer-specificity](https://www.nature.com/articles/s41467-025-61399-5), Kermani Nejad et al., 2025, asigna papeles a L2/3, L4 y L5 en predicción contextual y representaciones objetivo. Tiene una [corrección de octubre de 2025](https://www.nature.com/articles/s41467-025-65076-5). Es una motivación relevante; trasladarla a neuronas medidas exige explicitar quién predice, qué objetivo recibe y cómo se asigna crédito.

Otras fuentes: [Lotter et al. 2020](https://www.nature.com/articles/s42256-020-0170-9), [CPC](https://arxiv.org/abs/1807.03748), [SFA](https://www.ini.rub.de/PEOPLE/wiskott/Abstracts/WisSej2002.html), [predictive coding en grafos computacionales](https://arxiv.org/abs/2006.04182), [equilibrium propagation](https://pubmed.ncbi.nlm.nih.gov/28522969/).

**Arquitectura biológica, objetivo bioinspirado y regla local plausible no son equivalentes.** Usar BPTT para comparar arquitecturas primero. Las reglas locales requieren luego un experimento factorial grafo × regla; cambiar simultáneamente todas las piezas impide atribuir diferencias.

## C. Modelo de observación neuronal

Un estado recurrente no es fluorescencia. Partir de una tasa no negativa, filtrar calcio y calibrar gain/offset por célula:

$$c_{i,t+1}=e^{-\Delta t/\tau_i}c_{i,t}+(1-e^{-\Delta t/\tau_i})r_{i,t},\qquad \hat y_{i,t}=a_ic_{i,t}+b_i.$$

El código implementa un filtro exponencial con tau común configurable. Ajuste por célula, sincronización de sesiones y calibración completa siguen pendientes. El filtro es una aproximación, no toda la fisiología del indicador.

Para fluorescencia, usar error cuadrático normalizado por escalas fijadas solo con entrenamiento:

$$L_{calcium}=|I|^{-1}\sum_{i\in I}|O_i|^{-1}\sum_{t\in O_i}\frac{(\hat y_{it}-y_{it})^2}{2s_i^2}.$$

Es una verosimilitud gaussiana salvo constantes, no una afirmación de independencia temporal de residuos. La media por célula evita favorecer grabaciones largas. Considerar Huber como sensibilidad a outliers. La correlación no es suficiente como única pérdida porque ignora escala y offset.

Para **recuentos de spikes**, lambda - k log(lambda) es Poisson salvo constante. Actividad deconvolucionada fraccional no es un recuento observado: su uso exige declararla quasi-likelihood y validar ruido. Fluorescencia normalizada negativa no admite esa interpretación. El código rechaza incompatibilidades en lugar de ocultarlas.

Evaluar en vídeos reservados, por fiabilidad neuronal, y distinguir lectura por identidad de una proyección libre de población. Esta última puede ajustar respuestas aunque ninguna neurona artificial corresponda a su célula biológica. Ajustar todos los pesos a la fisiología constituye otro estudio supervisado.

## Regularización y falsos descubrimientos

Empezar con clipping y estabilidad. No imponer sparsity, tasas objetivo y decorrelación para presentar después esas mismas propiedades como hallazgos. Todo prior de pesos, actividad o constante temporal necesita ablación. La contracción implementada es una condición numérica que puede excluir dinámicas relevantes; no atribuirla al conectoma.
