# CTNet-Neuro

## CTNet V7-Neuro

CTNet V7-Neuro es una arquitectura neuronal holónica n-dimensional.

Cada nodo mantiene una carta local completa del sistema:

- estado dinámico
- estructura simbólica
- cuerpo
- tejido
- contexto
- ejecutivo local
- memoria
- offline/replay
- régimen
- verificador
- Babel local
- métricas

La globalidad no se obtiene mediante un pipeline central, sino por recirculación entre cartas locales. El atlas aplica transformaciones vectoriales Q_ij entre nodos, reconstruye un estado global X_hat, calcula errores de cierre y produce una lectura por consenso.

## Componentes principales

- ctnet_neuro.v7_neuro.NDimensionalAtlas: atlas n-dimensional con nodos, topología, transformaciones Q_ij, reconstrucción global y errores de cierre.
- HolonicNodeState: firma completa de cada nodo-carta local.
- V7NeuroNode: actualización local con dinámica, cuerpo, tejido, contexto, ejecutivo, memoria, Babel local y verificador.
- LocalCoherenceTensor: coherencia local, vecinal, corporal, tisular, contextual, memorial y offline.
- LocalMemory: memoria topológica acotada, compresión e invariantes.
- OfflineReplay: replay, consolidación y pruning con efecto real sobre estado.
- CTNetV7NeuroOrchestrator: ciclo ejecutable con mundo, cuerpo, tejido, contexto, atlas, reconstrucción, consenso y métricas.

## Ejecución rápida

    python -m examples.run_v7_neuro_demo

## Tests principales

    python -m pytest -q tests/test_v7_neuro_state.py tests/test_v7_neuro_node.py tests/test_v7_neuro_orchestrator.py tests/test_v7_neuro_no_pipeline.py

## Estado validado

- 11 tests pasan correctamente.
- La demo ejecuta 20 steps sin errores.
- La salida imprime coherencia global/local, error de cierre, energía/estrés corporal, excitación/inhibición tisular, modos ejecutivos, offline/replay y readout de consenso.
- La compatibilidad histórica del repo se conserva.
