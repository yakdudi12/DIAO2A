# TP3 - Perceptrón simple y multicapa

El trabajo implementa perceptrones simples (lineal y sigmoide) y un perceptrón multicapa
para dos encargos de CompanyX: destilar un modelo de detección de fraude (ejercicio 1) y
clasificar dígitos escritos a mano de 28×28 píxeles (ejercicios 2 y 3). Las redes,
la retropropagación, los optimizadores y el aumento de datos son propios; pandas y
scikit-learn se usan únicamente para cargar datos, particionar y calcular métricas.

## Ejecución

Desde la raíz del repositorio:

```bash
uv sync
uv run --with jupyter jupyter lab TP3/
```

También se pueden abrir los notebooks en VS Code y seleccionar `.venv/bin/python` como
kernel. Cada notebook busca los datos en `TP3/data/`, por lo que funciona tanto abierto
desde `TP3/` como desde la raíz.

| Notebook | Contenido |
|---|---|
| `ejercicio1.ipynb` | TinyModel para fraude: perceptrón lineal vs. sigmoide, generalización y umbral |
| `ejercicio2.ipynb` | MLP sobre `digits.csv`: arquitectura, tasa de aprendizaje y optimizador |
| `ejercicio3.ipynb` | MLP con `more_digits.csv`: aumento de datos, ponderación y sobremuestreo |

Los CSV ya están descomprimidos en `TP3/data/`. `data and documentation.zip` es el
paquete original de la cátedra. El enunciado menciona `more_data_digits.csv`; el archivo
entregado se llama `more_digits.csv`.

## Ejercicio 1: fraude (knowledge distillation)

TinyModel es un perceptrón simple de nueve entradas que aprende a imitar
`big_model_fraud_probability` con MSE. La etiqueta `flagged_fraud` no se usa para
entrenar: sólo estratifica las particiones y evalúa la detección. Las entradas se
estandarizan con las estadísticas de entrenamiento.

- Comparación con todo el dataset: la sigmoide llega a MSE 0,0119 frente a 0,0261 del
  lineal y 0,0915 del baseline que predice la media. Se elige la sigmoide.
- Generalización: partición estratificada 70/15/15, selección de tasa de aprendizaje y
  época por MSE de validación (lr = 0,03, época 472).
- Umbral: se marca fraude si `TinyModel >= 0,8822`, elegido por máximo F1 en validación.

## Ejercicios 2 y 3: dígitos

`DigitMLP` (en `mlp.py`) usa capas ocultas ReLU, salida softmax, entropía cruzada y
minibatches vectorizados. Soporta SGD, momentum, RMSProp y Adam.

- Ejercicio 2: partición 80/20 de `digits.csv` y grilla de 3 arquitecturas × 3 tasas ×
  4 optimizadores (36 configuraciones, 15 épocas). La configuración elegida es (128, 64),
  momentum y lr = 0,1. `digits.csv` no contiene ningún 8, lo que limita el resultado.
- Ejercicio 3: `digits.csv` y `more_digits.csv` se combinan sin imágenes repetidas
  (24.501 imágenes) y se separa 15 % para validación. Se prueban aumento de datos
  (desplazamiento ±2 px, rotación ±10°, zoom ±10 %), ponderación de la pérdida por clase
  y sobremuestreo del 5 y del 8. Sólo el aumento de datos mejora la validación.

En ambos ejercicios `digits_test.csv` se usa una única vez, al final, como si fueran datos
de producción.

## Estructura

```text
TP3/
├── ejercicio1.ipynb
├── ejercicio2.ipynb
├── ejercicio3.ipynb
├── perceptron.py               #Perceptrón simple (ejercicio 1)
├── activations.py              #identity, relu, sigmoid y sus derivadas
├── optimizer.py                #SGD, Momentum, RMSProp y Adam
├── mlp.py                      #DigitMLP, fit, aumento de datos y pesos por clase (ej. 2 y 3)
├── Enunciado TP3 - 2Q 2026.pdf
├── data and documentation.zip  #Paquete original de la cátedra
├── data/
│   ├── fraud_dataset.csv
│   ├── fraud_dataset_documentation.pdf
│   ├── digits.csv
│   ├── more_digits.csv
│   ├── digits_test.csv
│   └── results/                #Salidas de other_implementations
├── figures/                    #GIFs del ejercicio 1
├── figures_ej2/
├── figures_ej3/
├── logs/                       #Logs de las corridas en el cluster
└── other_implementations/      #Primera iteración en scripts .py (para cluster, con sbatch)
```

## Resultados

| Ejercicio | Modelo | Resultado en test |
|---|---|---|
| 1 | Perceptrón sigmoide, umbral 0,8822 | MSE vs. BigModel 0,0102; precision 0,922, recall 0,817, F1 0,866 |
| 2 | MLP (128, 64), momentum, lr = 0,1 | Accuracy 86,62 %, F1 macro 0,820 |
| 3 | Mismo MLP + más datos + aumento de datos | Accuracy **98,12 %** (objetivo ≥ 98 %), F1 macro 0,981 |

En el ejercicio 3, sólo agregar `more_digits.csv` lleva la accuracy de 86,62 % a 96,96 %
sin cambiar el modelo; el aumento de datos aporta el resto hasta 98,12 %.

Las figuras de cada notebook se guardan en:

- `figures/`: `ajuste_lineal_vs_sigmoide.gif` y `evolucion_pesos_perceptrones.gif`
- `figures_ej2/`: distribución de clases, grilla de hiperparámetros, matriz de confusión
  y análisis de los 8 en test
- `figures_ej3/`: curvas de validación, comparación de técnicas y evaluación final
