# CTNet V7-Neuro

CTNet V7-Neuro es una arquitectura holónica n-dimensional: cada nodo mantiene una carta local completa del sistema (dinámica, simbólica, cuerpo, tejido, contexto, ejecutivo, memoria, offline, régimen y verificación), y la globalidad se reconstruye por recirculación entre cartas.

## Dinámica vectorial local
- `HolonicNodeState.dynamic` es un vector serializable `list[float]`, no un escalar.
- Cada nodo actualiza todas sus componentes dinámicas en cada paso usando señal local `u/p`, drive de mundo y proyección de vecinos.
- La semántica local y la lectura Babel derivan del estado vectorial, no de un pipeline lineal central.

## Q_ij como transformación de carta
- `Q_ij` es una transformación vectorial (matriz determinista con permutación + escala), aplicada entre cartas locales.
- La recirculación vecinal usa `Q_ij n_j` para proyectar el estado de cada vecino `j` en el marco local de `i`.

## Reconstrucción global
- `reconstruct_global()` devuelve un vector global `X_hat` de dimensión `d`.
- La reconstrucción se calcula como promedio de cartas inversamente proyectadas al espacio global.
- `local_closure_errors()` compara cada carta local contra `Q_i X_hat` en error medio absoluto por componente.

## Causalidad entre nodos
- Cambios en un nodo remoto alteran la recirculación de sus vecinos.
- Esa perturbación modifica `X_hat`, errores de cierre y el consenso global de lectura.
- La salida global emerge por consenso ponderado entre nodos; no por concatenación ni por módulo soberano único.

## Invariantes de diseño
- Cada nodo contiene firma completa V7-Neuro.
- Coherencias siempre acotadas en `[0,1]`.
- Memoria local de capacidad fija (sin crecimiento ilimitado).
- Offline/replay modifica consolidación, pruning y masa semántica local.
- Estabilidad numérica: sin NaN ni errores de cierre infinitos en corridas largas.

## Componentes
- `HolonicNodeState`: firma local completa de cada nodo.
- `NDimensionalAtlas`: topología `A`, transformaciones `Q_ij`, recirculación, reconstrucción global y errores de cierre.
- `LocalCoherenceTensor`: coherencias locales y entre vecinos con salida acotada `[0,1]`.
- `LocalMemory`: memoria topológica de capacidad fija, con compresión y masa invariante.
- `LocalBabel`/lector/verificador local: lectura estructural por nodo, frontera no verificada y consenso global.
- `ExecutiveLocal`: modos `intent/explore/decide/reflect/stabilize/sleep/repair` que alteran tasas, compresión, recirculación, verificación y offline.
- `BodyBridge`, `TissueBridge`, `ContextBridge`: proyección local desde estado corporal, tejido y contexto existentes.
- `OfflineReplay`: replay/consolidación/pruning con efecto real sobre masa semántica y estabilidad.

## u/p
`u/p` representa inyección local por nodo (input-perception) usada en cada `step` del atlas junto con señales del mundo y vecinos.

## Métricas impresas en demo
- step
- global coherence
- mean local coherence
- closure error
- body energy/stress
- tissue excitation/inhibition
- executive modes
- offline status
- readout

## Ejecutar demo
```bash
python -m examples.run_v7_neuro_demo
```

## Ejecutar tests
```bash
pytest -q tests/test_v7_neuro_state.py tests/test_v7_neuro_node.py tests/test_v7_neuro_orchestrator.py tests/test_v7_neuro_no_pipeline.py
```
