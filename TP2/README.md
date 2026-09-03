# TP2 - Algoritmos genéticos con triángulos

El programa aproxima una imagen mediante una cantidad fija de triángulos RGBA sobre un
canvas blanco. La implementación del algoritmo genético es propia; Pillow, SciPy,
scikit-image y CuPy se usan únicamente para imágenes, métricas y aceleración numérica.

## Ejecución

Desde la raíz del repositorio:

```bash
uv sync
uv run --with jupyter jupyter lab TP2/tp2.ipynb
```

También se puede abrir `TP2/tp2.ipynb` en VS Code y seleccionar `.venv/bin/python` como
kernel.

La imagen se configura mediante `IMAGE_NAME` y debe existir en
`TP2/imagenesprueba/`. La cantidad de triángulos se configura con `TRIANGLE_COUNT`.

## Estructura

```text
TP2/
├── tp2.ipynb
├── imagenesprueba/
└── tp2_ga/
    ├── engine.py
    ├── fitness.py
    ├── io.py
    ├── operators.py
    └── render.py
```

## Operadores disponibles

- Selección: elite, roulette, universal, boltzmann, ranking,
  tournament_deterministic y tournament_probabilistic.
- Cruza: one_point, two_point y uniform.
- Mutación: gene, multigene y non_uniform.
- Supervivencia: additive y exclusive.

La población inicial se genera de forma completamente aleatoria. La imagen objetivo se
utiliza sólo durante la evaluación del fitness.

## CUDA

En Linux se instala `cupy-cuda12x`. Si CuPy puede abrir un dispositivo CUDA, el
rasterizado y el fitness se calculan en GPU. En caso contrario, el programa continúa en
CPU automáticamente.

El notebook muestra el backend seleccionado antes de comenzar. La primera corrida CUDA
incluye el costo de compilación del kernel.

## Resultados

La última sección del notebook crea `TP2/resultados/` con:

- `reconstruccion_final.png`
- `triangulos_finales.json`
- `metricas.json`

El JSON de triángulos contiene la enumeración completa solicitada por el enunciado.
