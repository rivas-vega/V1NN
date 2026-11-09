# Continuidad del proyecto BIONN / V1NN

- Leer `docs/BITACORA.md`, `docs/PROTOCOL.md`, `docs/RESEARCH_STATE.md` y el estado de Git antes de continuar.
- Trabajar de forma autónoma dentro de la petición del usuario; conservar historial
  y cambios ajenos. No hacer force push.
- Añadir una entrada a la bitácora tras cada experimento o cambio significativo:
  configuración, revisión, datos, métricas, fallos, decisión y límites. Mantener
  resultados negativos y separar observación, hipótesis y trabajo pendiente.
- No afirmar ventaja del conectoma a partir de un smoke test, una semilla o una
  mejora del error de entrenamiento. La validación del piloto sirve para ajustar;
  no presentarla como test final. Mantener comparaciones emparejadas.
- Antes de gastar GPU, comprobar `reports/compute_ledger.json` y el consumo de
  Modal. El presupuesto inicial autorizado el 2026-11-09 es 50 USD acumulados para
  esta primera ronda, no 50 USD por sesión. No contar créditos como permiso para
  ampliar uso. Conservar margen; registrar también campañas fallidas.
  El usuario autorizó después un máximo condicional de 75 USD solo si es
  estrictamente necesario para esta primera fase. Mantener 50 USD como límite
  operativo mientras baste. Antes de activar la ampliación, conciliar gasto y
  reservas y documentar qué experimento indispensable no cabe en 50 USD; no
  ampliarlo por comodidad ni por reservas pendientes de conciliar.
- Si la facturación no puede consultarse, no presentar una estimación como coste
  real. Anotar una reserva conservadora por campaña y detener nuevas campañas si
  el margen restante no es verificable. No crear bucles pagados ilimitados.
- Credenciales, datos crudos y checkpoints quedan fuera de GitHub. Publicar solo
  código, documentación y reportes pequeños sin secretos.
- Verificar con `pytest -q`, pruebas de integración pertinentes y `git diff --check`.
  Regenerar `MANIFEST_SHA256.json` al cambiar sus archivos cubiertos.
- Las campañas de Modal pueden continuar con `--detach`. No confundir esa
  persistencia con una sesión de agente permanente o con automatizaciones ya creadas.
