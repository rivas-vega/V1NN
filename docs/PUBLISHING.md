# Estado de publicación

El acceso de escritura a `rivas-vega/V1NN` quedó comprobado el 9 de noviembre de 2026: la integración GitHub creó el primer commit `6bdae1dc6e23470e5d552fe7b60512b23e312b51` en `main`. El error HTTP 403 de la tentativa anterior quedó resuelto. Esta revisión incorpora el paquete del prototipo, la documentación y las pruebas, conservando ese historial y sin force push.

Antes de publicar se verificó el manifiesto SHA256 del paquete y se volvieron a ejecutar sus pruebas: **20 tests superados**. Esto valida las comprobaciones de software existentes; no constituye entrenamiento ni evaluación del conectoma MICrONS. Los informes de smoke tests conservan sus resultados originales, incluidos los casos que no superan las referencias triviales.

## Trabajar desde una sesión propia autorizada

El repositorio ya tiene historial. Para continuar, clonar en lugar de inicializar otro historial paralelo:

```bash
git clone https://github.com/rivas-vega/V1NN.git
cd V1NN
python -m pip install -e '.[test,data]'
pytest -q
```

Para cambios posteriores, crear una rama, revisar los archivos y enviar un commit sin sobrescribir el historial:

```bash
git switch -c nombre-del-cambio
git status --short
git add <archivos-revisados>
git commit -m "Describe the change"
git push -u origin nombre-del-cambio
```

No pegar tokens en scripts ni documentación. No publicar datos crudos, credenciales ni checkpoints sin verificar sus condiciones de uso. `MANIFEST_SHA256.json` cubre los otros 21 archivos de esta entrega, no el propio manifiesto; se debe regenerar si cambian los archivos cubiertos.
