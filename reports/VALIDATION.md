# Validación ejecutada

Fecha: 9 de noviembre de 2026. Python 3.13.5, PyTorch 2.10.0+cpu, CPU.

## Tests

`PYTHONPATH=src pytest -q`: **20 passed**. Se comprueban orientación pre→post, signos E/I, gradientes, causalidad, memoria sin aristas, routing idéntico, filtros sin memoria, alineación futura, datos ausentes, compatibilidad de pérdidas e invariantes de rewiring. La importación se prueba con fixtures sintéticos e IDs largos; no es una validación de la descarga real.

## Integración sintética

64 nodos, 303 aristas, semilla 7, 25 actualizaciones por objetivo. **No MICrONS.**

```bash
PYTHONPATH=src python scripts/smoke.py --steps 25 --objective motion --output outputs/smoke_motion.json
PYTHONPATH=src python scripts/smoke.py --steps 25 --objective sensory_ssl --output outputs/smoke_ssl.json
```

| Prueba | Error held-out de la red | Referencia trivial |
| --- | --- | --- |
| Movimiento | 1.57416689 | Velocidad cero: 1.53125000 |
| SSL sensorial | 0.00257047 | Persistencia: 0.00060486 |
| SSL sensorial | 0.00257047 | Media entrenamiento: 0.00063376 |

**Ninguna prueba supera su referencia trivial.** La pérdida de entrenamiento disminuye, pero esto no demuestra computación útil. Son comprobaciones pequeñas de ejecución y gradientes, no evidencia de ventaja ni refutación de un modelo. Se adjuntan ambos JSON exactos en este directorio.

## No realizado

No se ha entrenado MICrONS, obtenido trazas funcionales, probado superioridad de topología ni validado coste GPU. Faltan runner multisemilla y baselines completos. No se ha reproducido FLYNN ni el estudio de Lappalainen. El bloqueo inicial de publicación GitHub por HTTP 403 quedó resuelto; véase docs/PUBLISHING.md. Los 20 tests se volvieron a ejecutar satisfactoriamente antes de la publicación.
