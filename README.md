# V1NN

**¿Qué aporta la conectividad medida de V1 cuando el modelo no puede delegar la tarea en un encoder preentrenado?**

V1NN es la rama experimental de córtex visual de ratón del programa BIONN. Estudia RNN dispersas restringidas por MICrONS. No simula un cerebro completo ni presupone que una topología biológica sea mejor.

## Decisión científica

El experimento principal **no es JEPA**. Empieza con estimación supervisada de movimiento a partir de vídeo, filtros espaciales fijos, una RNN del conectoma y una lectura lineal. Las etiquetas proceden del generador de escenas, no de un modelo visual. La predicción autosupervisada de señales sensoriales futuras es una comparación secundaria sin teacher. La correspondencia con actividad neuronal es un tercer análisis separado.

El precedente directo es [Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3): un modelo restringido por conectividad del sistema visual de la mosca, optimizado para detectar movimiento y contrastado con respuestas neuronales. No se presupone que sus resultados se transfieran a V1 de ratón.

## Estado real

| Componente | Estado |
| --- | --- |
| RNN dispersa, signos de Dale, entrada fija y lectura lineal | Implementado |
| Nulos dirigidos conservando grados, grupos y distancias discretizadas | Implementado, sin garantía de muestreo uniforme |
| Pérdidas de movimiento, predicción sensorial y observación neuronal | Implementadas |
| Importador MICrONS v1718 con verificación de hashes | Descarga real verificada: 1.357 neuronas, 78.953 parejas dirigidas |
| Tests unitarios e integración sintética | 44 tests superados; integración con nulos E/I y espaciales auditados |
| Piloto temporal de desarrollo | GRU/CNN aprenden; matriz espacial cruzada 3×3: EPE medio −0,20 %, signos mixtos y MSE 0,24 % peor para MICrONS; sin ventaja robusta del cableado fino |
| MICrONS frente al nulo de grados, confirmación congelada | Ventaja pequeña en 2048 clips nuevos; esta confirmación no incluye el control E/I posterior |
| Validación funcional y benchmark confirmatorio completo | **No evaluados** |

Los smoke tests de 25 actualizaciones **no superan predictores triviales**. Son pruebas de ejecución y gradientes, no evidencia a favor o en contra del conectoma. Véase [el informe](reports/VALIDATION.md).

## Empezar

Python 3.10 o posterior. En Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,data]'
pytest -q
python scripts/smoke.py --steps 25 --objective motion --output outputs/motion.json
python scripts/smoke.py --steps 25 --objective sensory_ssl --output outputs/ssl.json
```

Por defecto se crea un grafo **sintético de 64 nodos**, no MICrONS. Para preparar el grafo real en un equipo con conexión:

```bash
python scripts/prepare_microns.py
python scripts/smoke.py --graph data/processed/column_v1718.npz --steps 25
```

El segundo comando sigue siendo una comprobación de ejecución. El importador rechaza hashes incorrectos y la red rechaza signos presinápticos desconocidos. No redistribuir datos crudos, credenciales ni checkpoints sin verificar condiciones de uso.

## Documentación

- [Bitácora de BIONN/V1NN](docs/BITACORA.md): resultados, hipótesis, retoques y decisiones acumuladas.
- [GPU y conexión a Modal](docs/MODAL.md): autenticación, campañas acotadas y recuperación de resultados.
- [Protocolo](docs/PROTOCOL.md): hipótesis, controles, métricas y criterios de continuación.
- [Pérdidas y bibliografía](docs/LOSSES.md): aprendizaje supervisado, SSL bioinspirado y observación neuronal.
- [Datos y arquitectura](docs/DATA.md): calidad, fronteras, procedencia e interfaz.
- [Hoja de ruta](docs/ROADMAP.md): implementación existente y trabajo pendiente.
- [Publicación](docs/PUBLISHING.md): estado de la subida y comandos de publicación.

`configs/pilot.json` describe una propuesta, **no una prerregistración externa ni un runner completo**. Arquitectura biológica, objetivo bioinspirado y algoritmo local plausible son tres conceptos distintos. El prototipo usa BPTT.

El nuevo piloto de desarrollo se ejecuta con:

```bash
python scripts/pilot.py --model gru --steps 500 --output outputs/gru.json
```

Usa validación para explorar y ajustar; no sustituye al protocolo completo ni a un
test confirmatorio. Ver la bitácora para resultados y limitaciones actuales.
