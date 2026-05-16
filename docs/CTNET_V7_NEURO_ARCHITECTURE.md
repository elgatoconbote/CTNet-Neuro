# CTNet V7-Neuro

CTNet V7-Neuro es una arquitectura holónica n-dimensional: cada nodo mantiene una carta local completa del sistema (dinámica, simbólica, cuerpo, tejido, contexto, ejecutivo, memoria, offline, régimen y verificación), y la globalidad se reconstruye por recirculación entre cartas.

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
