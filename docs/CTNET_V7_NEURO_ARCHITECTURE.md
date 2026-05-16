# CTNet V7-Neuro

CTNet V7-Neuro es una arquitectura holónica n-dimensional. Cada nodo contiene una carta local de la totalidad: estado dinámico, simbólico, cuerpo, tejido, contexto, ejecutivo, memoria topológica, offline/replay, régimen y verificador local.

## Principios
- No usa pipeline lineal V7->Neuro->respuesta.
- El estado global se reconstruye por recirculación entre cartas locales.
- La salida global emerge por consenso entre lectores/verificadores locales.

## Componentes
- `HolonicNodeState`: firma completa local.
- `NDimensionalAtlas`: topología A, transformaciones Q, reconstrucción y errores de cierre.
- `LocalCoherenceTensor`: C_local, C_neighbors, C_body, C_tissue, C_context, C_memory, C_offline, C_total y masa.
- `LocalMemory`: memoria topológica acotada (chart/reconstruction/recirculation) con compresión, consolidación y pruning.
- `LocalBabel` + `LocalVerifier`: lectura estructural local, frontera no verificada y chequeo de consistencia.
- `ExecutiveLocal`: modos intent/explore/decide/reflect/stabilize/sleep/repair con efecto real sobre actualización, compresión, recirculación, verificación y offline.
- `OfflineReplay`: replay/restoration/cleanup/consolidation/pruning local.

## u/p
u/p es señal local de actualización/proyección por nodo; modula el estado dinámico y la estructura simbólica en cada paso.

## Métricas impresas en demo
Paso, coherencia global, coherencia local media, error de cierre, energía/estrés corporal, excitación/inhibición tisular, modos ejecutivos activos, estado offline y readout generado.

## Ejecución
```bash
python -m examples.run_v7_neuro_demo
pytest -q tests/test_v7_neuro_state.py tests/test_v7_neuro_node.py tests/test_v7_neuro_orchestrator.py tests/test_v7_neuro_no_pipeline.py
```
