# Trabajo con GPU en Modal

El cliente opcional se instala con `python -m pip install -e '.[modal]'`.
Autenticar una vez con `modal token new`; completar el inicio de sesión en la
página que abre Modal. No pegar tokens en el chat ni guardarlos en este repositorio.
Puede elegirse una ubicación privada mediante `MODAL_CONFIG_PATH`. Mantener el
archivo con permisos 0600. [Autenticación oficial](https://modal.com/docs/cli/latest/token).

## Ejecución acotada

Primero pasar `pytest -q` y guardar la revisión de código en Git. El lanzador
rechaza árboles con cambios sin guardar para identificar exactamente el código.

```bash
modal run --detach scripts/modal_pilot.py \
  --models gru,static,linear_history,temporal_cnn --seeds 7,8,9 \
  --steps 500 --seconds 1200

modal run --detach scripts/modal_pilot.py \
  --graph microns --models connectome,no_edges,reset --seeds 7 \
  --steps 500 --seconds 1200
```

`--detach` mantiene la campaña en Modal si se pierde la conexión local. El bucle
de experimentos se ejecuta remotamente, de modo que no depende de que el cliente
envíe cada ensayo. Cada campaña usa una T4, un máximo de un contenedor, cero
reintentos automáticos, timeout de servidor de 1.800 segundos y un presupuesto
solicitado entre 60 y 1.500 segundos. Cada ensayo limita su entrenamiento a 300 s;
el informe distingue ensayos completos y truncados. Preparación y evaluación
pueden añadir tiempo al límite cooperativo; el timeout del servidor es la cota
final de ejecución de la función. El arranque y construcción de imagen se cobran
aparte cuando corresponda.

Resultados, configuración y checkpoints se guardan tras cada ensayo en el volumen
`bionn-v1nn-experiments`, en un directorio de campaña único. Un fallo queda reflejado
en `campaign.json`; no se reinicia automáticamente. El checkpoint actual se escribe
al terminar un ensayo, por lo que una interrupción brusca puede perder ese ensayo.
No se ha implementado reanudación automática de entrenamiento.

```bash
modal app list
modal volume ls bionn-v1nn-experiments
mkdir -p ./outputs/ID_DE_CAMPAÑA
modal volume get bionn-v1nn-experiments /ID_DE_CAMPAÑA ./outputs/ID_DE_CAMPAÑA
modal app stop ID_DE_APP
```

Referencias: [ejecución desconectada](https://modal.com/docs/guide/apps),
[volúmenes](https://modal.com/docs/guide/volumes) y
[timeouts](https://modal.com/docs/guide/timeouts).

`--alpha` controla la velocidad de actualización del estado, entre 0 y 1,
excluyendo 0. El valor original es 0,3. Cada informe conserva el valor usado;
modificarlo es una decisión experimental que debe registrarse en la bitácora.

Crear el directorio de descarga antes de `modal volume get` conserva los archivos
separados. Comprobar siempre el número de JSON y checkpoints recuperados.

## Presupuesto de esta ronda

Autorización inicial: **50 USD como máximo acumulado**, no por campaña. Verificar
el consumo antes de cada nueva campaña y reservar margen para imágenes, CPU,
memoria y almacenamiento. No programar ejecuciones recurrentes sin un presupuesto
acumulado persistente y una condición de parada.

Tarifas consultadas el 2026-11-09: T4 0,000164 USD/s; CPU 0,0000131 USD/core/s;
memoria 0,00000222 USD/GiB/s. Con 2 cores y 8 GiB solicitados, 30 minutos serían
aproximadamente 0,375 USD de recursos solicitados, sin incluir sobreconsumo,
arranque, imagen, red ni almacenamiento. Esto es una estimación, no una factura ni
un límite de facturación. [Tarifas oficiales](https://modal.com/pricing).

Modal ofrece límites de uso y gasto en su panel de facturación; no se modifica
globalmente la cuenta porque puede contener otros proyectos. Si el usuario quiere
un corte de facturación global, puede configurar el límite allí. Los presupuestos
por Environment dependen del plan. [Presupuestos oficiales](https://modal.com/docs/guide/budgets).

La persistencia de la campaña GPU y las revisiones del agente son mecanismos
distintos. Por petición del usuario, Codex tiene una revisión de este chat cada
dos horas durante la noche del 9 al 10 de noviembre de 2026, con fin a las 08:00
Europe/Paris. Esa configuración vive en la aplicación, no en este repositorio.
El ordenador debe permanecer encendido y la aplicación abierta para acceder a
los archivos locales. Consultar `RESEARCH_STATE.md` y el estado local antes de
continuar; las campañas ya lanzadas con `--detach` permanecen en Modal.
[Requisitos de las tareas locales](https://learn.chatgpt.com/docs/automations?surface=app).

`--train-size`, `--batch` y `--lr` permiten diagnósticos controlados. Sus cotas
son 1..4096 clips, 1..128 ejemplos por lote y una tasa finita entre 0 y 0,1
(excluyendo 0). El informe distingue el error del lote durante entrenamiento
del error sobre todos los clips después de la última actualización.
