# CTNet V7-Neuro

CTNet V7-Neuro es una arquitectura holónica n-dimensional: cada nodo mantiene una carta local completa del sistema y evoluciona como estado vectorial local, no como escalar.

## Invariantes de diseño
- Cada nodo tiene firma completa (`dynamic`, `symbolic`, `body`, `tissue`, `context`, `executive`, `memory`, `offline`, `regime`, `verifier`, `semantic_mass`, `local_babel`, `local_metrics`).
- `dynamic` es un vector serializable (`list[float]`) por nodo.
- `Q_ij` es transformación entre cartas (permuta componentes vectoriales).
- La reconstrucción global produce `X_hat` vectorial.
- La condición de cierre se evalúa por carta comparando `n_i.dynamic` contra `Q_i X_hat`.
- No existe módulo central soberano: la causalidad aparece por recirculación entre nodos.

## Componentes
- `HolonicNodeState`: firma local completa.
- `NDimensionalAtlas`: topología `A`, transformaciones `Q_ij`, recirculación, reconstrucción global y errores de cierre.
- `LocalCoherenceTensor`: coherencia local y entre nodos (`C_local`, `C_neighbors`, `C_body`, `C_tissue`, `C_context`, `C_memory`, `C_offline`, `C_total`, `mass`).
- `LocalMemory`: memoria topológica de capacidad fija con compresión.
- `LocalBabel`/lector/verificador local: lectura estructural por nodo y consenso global.
- `ExecutiveLocal`: modos `intent/explore/decide/reflect/stabilize/sleep/repair` que alteran dinámica local.
- `OfflineReplay`: replay/consolidación/pruning con efectos reales sobre estado.

## Causalidad entre nodos
En cada `step`, cada nodo `i` recibe recirculación de vecinos `j` mediante `Q_ij * n_j.dynamic`; al perturbar un nodo remoto, cambian `X_hat`, cierre y readout global.

## Ejecutar demo
```bash
python -m examples.run_v7_neuro_demo
```

## Ejecutar tests
```bash
pytest -q tests/test_v7_neuro_state.py tests/test_v7_neuro_node.py tests/test_v7_neuro_orchestrator.py tests/test_v7_neuro_no_pipeline.py
```
