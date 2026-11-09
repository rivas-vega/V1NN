# Datos, fronteras e interfaz neuronal

## Selección inicial

Priorizar la columna V1 de Minnie revisada manualmente, de aproximadamente 1.300 neuronas, frente a escoger arbitrariamente 5.000–10.000 células. La [documentación MICrONS](https://tutorial.microns-explorer.org/proofreading.html) diferencia errores de splits/merges y revisión de axones y dendritas. Un cero observado no siempre significa ausencia anatómica fiable.

El [notebook docente oficial](https://alleninstitute.github.io/Teaching_Connectomics/notebooks/microns_data_access_education.html) ofrece exportaciones estáticas. Se fija v1718 por reproducibilidad, no por ser la última versión. Migrar versiones exige comprobar identidades y no unir fechas distintas silenciosamente.

## Procedencia del importador

Repositorio: `AllenInstitute/connectomics_at_cosyne`. Commit: `513a6fe738f91ce179650f1cadf26d4a4728b21a`. Ruta: `docs/resources/data/`.

| Archivo | Git blob SHA1 esperado |
| --- | --- |
| v1718_cell_info.csv | 65bc15f566288ba763a25c7a6f91a94ab8a970ce |
| v1718_v1_column_synapses.feather | 48ba7fa1b6aee968ab53304a655461363deddbd1 |

El importador verifica hashes, calcula SHA256 y guarda manifiesto. Conserva IDs exactos como enteros/cadenas, nunca float, todos los nodos is_column incluidos aislados, y agrega sinapsis por pareja pre→post. No confunde tamaño sináptico con número. Reporta autapsis sin eliminarlas automáticamente.

**Estado:** referencias remotas comprobadas, lógica probada con fixtures. La descarga real no se completó en el entorno de desarrollo. Faltan auditoría del grafo importado, grados, signos y ejecución sobre MICrONS. Los recuentos exactos deben surgir del manifiesto, no copiarse de un ejemplo.

## Conectoma abierto e incompleto

Una columna revisada no es un circuito cerrado. Faltan entradas talámicas, conexiones laterales y feedback, y ramas salen del volumen. El archivo de conexiones internas **no permite estimar por sí solo qué fracción de toda la entrada fisiológica retiene cada célula**.

Estrategia propuesta: núcleo con calidad conocida; halo de vecinos cuando exista conectividad fiable; sensibilidad a fronteras, filtros y subvolúmenes. Registrar calidad de axones presinápticos, células añadidas y aristas recuperadas. Mantener la misma política de cierre para grafos medidos y nulos.

Un input externo compartido de dimensión baja puede representar contexto desconocido. Un proceso libre por neurona y tiempo absorbería la tarea. Imputar aristas introduce una hipótesis adicional, no recupera cableado real. No hay una corrección universal que transforme la muestra en todo el córtex.

## Arquitectura implementada

`FixedRetina` calcula contraste espacial entre medias 3×3 y 5×5, canales ON/OFF y pooling 4×4: 32 características por fotograma, sin parámetros ni historial. Es un filtro diagnóstico inspirado en contraste local, no retina o LGN validada.

Un escalar no negativo por neurona sigue:

$$h_{t+1}=(1-\alpha)h_t+\alpha\,\operatorname{ReLU}(Wh_t+u(x_t)+b).$$

Solo existen parámetros recurrentes sobre parejas observadas. El signo corresponde a la **neurona emisora**. Los signos desconocidos provocan error hasta que exista una política explícita. La entrada promedia un pequeño conjunto fijo de señales por nodo, con ganancia positiva aprendida y sesgo; la lectura es lineal sobre el estado. No hay rutas directas desde los píxeles a la salida.

Los recuentos sinápticos se conservan, pero no fijan todavía los pesos. Una extensión comparará pesos libres, compartidos por tipo y priors de multiplicidad. No interpretar parámetros entrenados como mediciones electrofisiológicas.

## Estabilidad y recursos

Se impone max_i sum_j |W_ij| <= g < 1. Con ReLU 1-Lipschitz y fuga común, la actualización tiene cota 1-alpha+alpha*g < 1. Es una condición suficiente de contracción **impuesta**, no un hallazgo sobre el córtex. Puede limitar memoria o excluir actividad relevante; variar alpha/g con los mismos controles y reportar sensibilidad.

El código usa acumulación por arista, no matriz densa N×N. Aun así materializa mensajes batch×aristas y estados batch×tiempo×neuronas. No es un kernel GPU optimizado ni un simulador continuo. Perfilado, BPTT truncado y microsteps son extensiones pendientes; bytes de pesos no bastan para garantizar entrenamiento en una A100.

## Datos funcionales

Mantener conjuntos separados: neuronas EM, células registradas y matches verificados. No todas las reconstrucciones tienen traza y no todas las trazas son simultáneas. Guardar versión, sesión, tiempos de estímulo y adquisición, y preprocesado. Fuente: [MICrONS Consortium 2025](https://www.nature.com/articles/s41586-025-08790-w).

No combinar sesiones para inferir correlaciones de ruido. Movimiento, pupila, mirada y estímulo condicionan interpretación. El routing mediante campos receptivos debe calibrarse solo con entrenamiento. Coordenadas corticales transformadas no son coordenadas visuales. El registro de calcio requiere modelo causal de observación y techo de ruido estimable.

Guardar unidades de posiciones, grupos realmente disponibles, identidad, hashes y manifiestos separados para escenas y registros. No inventar capa cortical si falta la anotación. Revisar condiciones de uso antes de redistribuir cualquier modalidad.
