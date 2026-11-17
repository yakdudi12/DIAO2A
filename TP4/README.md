# TP4 - Métodos de aprendizaje no supervisado

El trabajo aplica métodos no supervisados a dos problemas. El ejercicio 1 ayuda a una
agencia de marketing a elegir creadores de contenido para cuatro clientes (refugio de
perros, ropa de fútbol, belleza y materiales de arte) con PCA, una red de Kohonen y la
regla de Oja. El ejercicio 2 usa redes de Hopfield como memoria asociativa: primero con
letras de 5×5 y después para recuperar códigos QR deteriorados de las cartas de un
restaurante. PCA, Kohonen, Oja y Hopfield son propios, escritos con numpy; scikit-learn
se usa sólo para validar el PCA y `zxing-cpp` para leer los QR con un decodificador real.

## Ejecución

Desde la raíz del repositorio:

```bash
uv sync
uv run --with jupyter jupyter lab TP4/
```

También se pueden abrir los notebooks en VS Code y seleccionar el Python de `.venv` como
kernel. Los notebooks agregan `TP4/` al path y buscan los datos en `TP4/data/`, por lo que
funcionan tanto abiertos desde `TP4/` como desde la raíz.

| Notebook | Contenido |
|---|---|
| `ejercicio1.ipynb` | Campañas de marketing: EDA, PCA, Kohonen, índice con Oja y selección de creadores |
| `ejercicio2.ipynb` | Hopfield: letras de 5×5 (capacidad, ruido, espurios) y recuperación de QR |

`zxing-cpp` ya está en `pyproject.toml`, así que `uv sync` lo instala. En el ejercicio 2,
la búsqueda de todos los puntos fijos recorre los 2²⁵ ≈ 33 millones de estados de la red
de letras, y es la celda más pesada del notebook.

## Ejercicio 1: campañas de marketing

Datos: `data/creadores.csv` (50 creadores). Se excluyen los identificadores y las cuatro
columnas que son fórmulas exactas de otras, y quedan 27 variables. Las cinco de escala
(`followers`, `average_reach`, `average_impressions`, `average_video_views`, `price_usd`)
pasan a log10 y todas se estandarizan. Kohonen y Oja usan el mismo preprocesamiento.

- **1.1 EDA y PCA:** PCA sobre la matriz de correlación con `np.linalg.eigh`. PC1 explica
  ≈ 52 % y es el eje alcance masivo ↔ profundidad (correlación 0,93 con log10 de
  seguidores). PC2, PC3 y PC4 separan arte, fútbol y maquillaje. Kaiser y el 80 %
  acumulado coinciden en 6 componentes (≈ 82 %).
- **1.2 Kohonen:** grilla 5×5, η₀ = 0,5, radio 2,5 → 0,5, 100 épocas, elegida con un
  barrido de tamaño, radio, tasa, inicialización y épocas (5 semillas cada uno). Explica
  ≈ 72 % de la varianza con error topográfico nulo. La matriz U y los planos de
  componentes delimitan cinco regiones: perros (comunidad y profundidad), nicho profundo,
  arte, maquillaje masivo y masivo/descubrimiento.
- **1.3 Oja:** neurona lineal con η = 10⁻³ y 200 épocas. Converge a PC1
  (|cos| ≈ 0,9999, Spearman ≈ 1 con la proyección). El índice describe qué tan masivo o
  profundo es un creador y no sirve como ranking universal: su signo es arbitrario y no
  contiene el objetivo de la campaña.
- **1.4 Decisión:** `campaigns.py` define el costo por interacción de intención
  (guardados, comentarios y visitas al perfil sobre el alcance en Argentina, ponderado por
  el % de la temática del cliente). Para fútbol se recomienda priorizar descubrimiento con
  audiencia argentina. Se comparan tres estrategias: top seguidores del catálogo (la
  actual), top seguidores de la temática y temática + costo por intención (la propuesta).

## Ejercicio 2: Hopfield

Red discreta (±1) con actualización asincrónica y energía E = −½ sᵀWs.

- **2.2 Letras de 5×5:** entre los 14.950 grupos de 4 letras se eligen G, R, T, V, con
  solapamiento máximo 0,12, y se comparan con J, O, X, L y H, M, N, W. Se mide la capacidad
  real con 200 grupos al azar por tamaño, la recuperación con ruido y se enumeran todos los
  puntos fijos: 4 letras, 4 inversos y 56 espurios (28 mezclas de tres letras).
- **2.1 Códigos QR:** cinco QR de 37×37 (N = 1369) en `data/hopfield/`, con corrección de
  errores nivel L. Comparten el 70 % de los píxeles, así que con Hebb ni los originales son
  puntos fijos. Se usa la regla de la pseudoinversa. Se simulan cuatro deterioros (ruido
  aleatorio, esquina rota, mancha y rayones) y el éxito se mide con el lector `zxing-cpp`
  sobre el QR recuperado. También se analizan iteraciones, energía y limitaciones del caso
  de uso.

## Estructura

```text
TP4/
├── ejercicio1.ipynb
├── ejercicio2.ipynb
├── preprocessing.py            #Carga, selección de variables y estandarización (ej. 1)
├── eda.py                      #Resúmenes del análisis exploratorio
├── pca.py                      #PCA con numpy, bastón roto y criterios de retención
├── kohonen.py                  #Mapa autoorganizado, métricas y barrido de hiperparámetros
├── oja.py                      #Regla de Oja
├── campaigns.py                #Métricas de campaña y comparación de estrategias
├── hopfield.py                 #Red de Hopfield, puntos fijos y experimentos de recuperación
├── letters.py                  #Alfabeto 5×5, elección de letras y capacidad
├── noise.py                    #Ruido y daños (esquina rota, mancha, rayones)
├── qr.py                       #Carga de QR como patrones ±1 y lectura con zxing-cpp
├── plots.py                    #Gráficos de ambos ejercicios
├── SIATP42026.pdf              #Enunciado
├── TP4_presentacion.pdf        #Presentación
└── data/
    ├── creadores.csv
    └── hopfield/               #Los cinco QR de las cartas (.png)
```

## Resultados

Los notebooks no guardan archivos: tablas y gráficos quedan en las salidas de cada celda.

| Ejercicio | Resultado |
|---|---|
| 1.1 PCA | PC1 ≈ 52 % (masivo ↔ profundo); 6 componentes retienen ≈ 82 % |
| 1.2 Kohonen | 5×5, ≈ 72 % de varianza explicada, error topográfico nulo, cinco regiones |
| 1.3 Oja | \|cos(w, PC1)\| ≈ 0,9999 con η = 10⁻³ |
| 1.4 Campañas | La propuesta tiene el menor costo por intención en los cuatro clientes |
| 2.2 Letras | G, R, T, V: recuperación perfecta hasta 8 % de ruido, 99 % con 12 %, 90 % con 20 % |
| 2.1 QR | Con la red, se lee el 100 % hasta 30 % de ruido y el 98 % con 40 %; sin la red, ninguno desde 5 % |

Elegir por seguidores da los mismos tres creadores para todos los clientes. Para el
refugio y para arte ninguno es de la temática, y cada interacción relevante cuesta ≈ 17 y
≈ 14 USD, contra 0,9 y 1,7 USD de la propuesta.

En los QR, la red converge en una sola pasada hasta 30–40 % de daño y corrige exactamente
los píxeles dañados. Su principal riesgo es silencioso: con 45 % de ruido devuelve otro
menú en el 10 % de las consultas, y ante un QR desconocido siempre responde con uno de los
cinco guardados.
