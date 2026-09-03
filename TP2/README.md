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
- Cruza: one_point, two_point, uniform y region.
- Mutación: gene, multigene, non_uniform y triangle.
- Supervivencia: additive y exclusive.

La configuración recomendada usa `region` y `triangle`. El cruce regional intercambia
grupos espacialmente coherentes; la mutación por triángulo traslada, escala, modifica
vértices, color y alpha, puede reemplazar un triángulo por uno pequeño y ocasionalmente
cambia el orden de capas. Los métodos exigidos por el enunciado siguen disponibles.

La inicialización guiada crea una mezcla de triángulos globales y locales. Sus centros se
muestrean con mayor probabilidad cerca de bordes y sus colores parten del objetivo. Puede
desactivarse con `target_guided_initialization=False` para experimentar con una población
uniforme sin información previa.

## Fitness y resolución

El error combina diferencia RGB L1 ponderada por detalle, diferencia de gradientes Sobel
y `1 - SSIM`. Los pesos espaciales y la escala de bordes dependen sólo del objetivo, por
lo que un individuo no puede reducir la penalidad introduciendo un borde extremo.

Se evalúan simultáneamente una escala reducida y la escala corriente. Por defecto sus
pesos son 25 % y 75 %. El notebook progresa de 48 a 84 y 120 píxeles, alcanzando la
resolución final al 65 % de la corrida. Es más costoso que evaluar sólo a 64 píxeles, pero
deja generaciones suficientes para optimizar detalles que existen en la salida final.

## CUDA

En Linux se instala `cupy-cuda12x`. Si CuPy puede abrir un dispositivo CUDA, el
rasterizado y el fitness se calculan en GPU. En caso contrario, el programa continúa en
CPU automáticamente.

El notebook muestra el backend seleccionado antes de comenzar. La primera corrida CUDA
incluye el costo de compilación del kernel.

## Resultados

La última sección del notebook crea `TP2/resultados_mejorados/` con:

- `reconstruccion_final.png`
- `triangulos_finales.json`
- `metricas.json`

El JSON de triángulos contiene la enumeración completa solicitada por el enunciado.
