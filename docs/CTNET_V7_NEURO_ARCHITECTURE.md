# CTNet V7-Neuro

CTNet V7-Neuro es una arquitectura holónica n-dimensional: cada nodo mantiene una carta local completa del sistema, y la globalidad emerge por recirculación entre cartas locales.

## Dinámica vectorial local
Cada nodo usa estado `dynamic: list[float]` (serializable en JSON) como vector local n-dimensional, no escalar.

## Q_ij como transformación de carta
`Q_ij` es una transformación vectorial (matriz determinista de permutación+escala) entre cartas locales. Cada nodo recibe proyecciones vecinales `Q_ij n_j`.

## Reconstrucción global
La reconstrucción global `X_hat` es un vector n-dimensional:
- `X_hat = (1/N) Σ_i Q_ii n_i`
- no es concatenación ni media escalar plana.

## Cierre local
Cada error de cierre compara carta local vs proyección local de global:
- `error_i = || n_i - Q_ii X_hat ||`

## Causalidad entre nodos
Una perturbación en nodo remoto altera `X_hat` y cambia coherencia/cierre/readout en pasos siguientes por recirculación topológica.

## Invariantes de diseño
- cada nodo contiene firma holónica completa;
- coherencias acotadas en `[0,1]`;
- memoria local con capacidad fija;
- offline/replay cambia consolidación/pruning y afecta estado local.

## Ejecutar demo
```bash
python -m examples.run_v7_neuro_demo
```

## Ejecutar tests
```bash
pytest -q tests/test_v7_neuro_state.py tests/test_v7_neuro_node.py tests/test_v7_neuro_orchestrator.py tests/test_v7_neuro_no_pipeline.py
```
