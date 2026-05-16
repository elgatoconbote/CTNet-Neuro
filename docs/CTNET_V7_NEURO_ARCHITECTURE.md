# CTNet V7-Neuro

CTNet V7-Neuro es una arquitectura n-dimensional holónica: cada nodo contiene una carta local de la totalidad, y la salida global emerge por coherencia y consenso entre cartas locales.

## Nodo holónico
Cada nodo modela localmente:
- estado dinámico
- estructura simbólica
- cuerpo/interocepción/propiocepción
- tejido neuronal/glial
- contexto/hipótesis
- ejecutivo local
- memoria topológica local
- offline/replay
- régimen
- verificador/lector local
- masa semántica
- Babel local

## u/p
u/p es la señal de actualización/proyección local que inyecta intención y control contextual en cada carta.

## Tensor de coherencia
`LocalCoherenceTensor` calcula coherencias locales y entre proyecciones: cuerpo, tejido, memoria, contexto, offline y vecindad.

## Memoria topológica
`LocalMemory` usa buffer acotado + reconstrucción local + compresión/recirculación + masa de invariantes.

## Babel local
Cada nodo tiene `LocalBabel` + `LocalVerifier`. La lectura global sale de `global_consensus(...)` y no de una concatenación plana.

## Cuerpo, tejido, contexto
Se usan puentes (`body_bridge.py`, `tissue_bridge.py`, `context_bridge.py`) para acoplar estado existente del repo y exponer variables locales requeridas.

## Ejecutivo local
Modos: `intent`, `explore`, `decide`, `reflect`, `stabilize`, `sleep`, `repair`.
Afectan tasa de actualización, apertura Babel, compresión de memoria, recirculación, prioridad de verificación y sesgo offline.

## Offline/replay
`OfflineReplay` aplica replay, restauración, cleanup, consolidación y pruning, alterando memoria y masa semántica local.

## Métricas impresas en demo
- step
- global coherence
- mean local coherence
- closure error
- body energy/stress
- tissue excitation/inhibition
- active executive modes
- offline/replay status
- generated readout

## Ejecutar demo
```bash
python -m examples.run_v7_neuro_demo
```

## Ejecutar tests
```bash
pytest -q tests/test_v7_neuro_state.py tests/test_v7_neuro_node.py tests/test_v7_neuro_orchestrator.py tests/test_v7_neuro_no_pipeline.py
```
