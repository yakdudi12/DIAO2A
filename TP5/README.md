# TP5 - Deep Learning: autoencoders

El trabajo implementa autoencoders desde cero, con `numpy` y retropropagación escrita a
mano, sin librerías de aprendizaje. El ejercicio 1 comprime las 32 letras de `font.h`
(imágenes binarias de 7×5) a un espacio latente de dos dimensiones y estudia un
*denoising autoencoder* sobre el mismo conjunto. El ejercicio 2 extiende el modelo a un
autoencoder variacional (VAE).

`re`, `PIL` y `matplotlib` se usan sólo para parsear el archivo de fuentes, rasterizar
imágenes y graficar; ningún algoritmo de aprendizaje depende de ellos.

## Ejecución

Desde la raíz del repositorio:

```bash
uv sync
uv run --with jupyter jupyter lab TP5/
```

También se pueden abrir los notebooks en VS Code y seleccionar el Python de `.venv` como
kernel. Los notebooks importan los módulos de `TP5/` desde la carpeta de trabajo, así que
el kernel tiene que arrancar en `TP5/` (es lo que hace VS Code por defecto).

| Notebook | Contenido |
|---|---|
| `ejercicio1.ipynb` | Autoencoder de las 32 letras de `font.h` y denoising autoencoder |
| `ejercicio2.ipynb` | VAE sobre símbolos matemáticos |
| `ejercicio2_propuesta_*.ipynb` | Propuestas alternativas de datos para el ejercicio 2 |

Todos los hiperparámetros y semillas están en `config.json`. Los barridos del ejercicio 1
ya calculados están en `results/`. Si esos archivos existen, el notebook los lee y corre en
unos 15 segundos. Si se borran, los recalcula (unas 130 redes, varios minutos).

## Ejercicio 1a: autoencoder con latente de 2 dimensiones

**Objetivo del enunciado:** aprender los 32 patrones de 7×5 en un espacio latente de dos
dimensiones con un error máximo de 1 píxel incorrecto.

**Modelo.** Codificador y decodificador simétricos, capa latente lineal y salida sigmoide.
La pérdida es la entropía cruzada binaria (los píxeles son 0/1). Su gradiente respecto del
logit es `predicción − objetivo`, sin el factor σ'(a) que frena el aprendizaje cuando la
salida satura con error cuadrático. Los gradientes se verifican con diferencias finitas
(error relativo máximo 8,8·10⁻⁹).

**Métrica de éxito.** Píxeles mal del *peor* patrón tras umbralizar la salida en 0,5.

### Búsqueda de arquitectura y optimizador

Grilla de 3 arquitecturas del codificador × 2 activaciones × 3 optimizadores, 3000 épocas
*full-batch* y 5 semillas por configuración. Las tasas de aprendizaje son SGD 0,03,
Momentum 0,005 y Adam 0,005. La tabla muestra la fracción de semillas que llega a ≤ 1 píxel
de error en el peor patrón:

| Codificador | Activación | SGD | Momentum | Adam |
|---|---|---|---|---|
| 35-16-2 | tanh | 0,0 | 0,0 | 0,4 |
| 35-16-2 | relu | 0,0 | 0,4 | 0,4 |
| 35-32-16-2 | tanh | 0,2 | 1,0 | 1,0 |
| 35-32-16-2 | relu | 0,8 | 1,0 | 1,0 |
| 35-64-32-2 | tanh | 0,8 | 1,0 | 1,0 |
| 35-64-32-2 | relu | 1,0 | 1,0 | 1,0 |

- **Arquitectura.** Con una sola capa oculta de 16 unidades ninguna combinación aprende los
  32 patrones de forma confiable. Con dos capas ocultas casi todas llegan a 0 píxeles mal.
  El límite no es el cuello de botella de 2 dimensiones sino la expresividad del
  decodificador.
- **Optimizador.** Adam logra 5 de 5 semillas en las cuatro configuraciones de dos capas y
  es el más rápido (mediana de 200 a 500 épocas hasta ≤ 1 píxel). Momentum necesita entre 2
  y 5 veces más épocas. SGD es el más frágil.
- **Activación.** ReLU converge antes que tanh en las arquitecturas profundas. Igual se
  elige tanh para el modelo final porque su salida acotada da un espacio latente más
  compacto y fácil de leer.
- **Línea de base lineal.** PCA con 2 componentes, equivalente a un autoencoder lineal,
  deja hasta 15 píxeles mal en el peor patrón (7,2 de media). Sin no linealidad, las dos
  dimensiones sí son una restricción real.

### Modelo final

Codificador 35-32-16-2 con tanh, Adam (lr = 0,005), 6000 épocas y semilla 0. Reconstruye
**los 32 patrones con 0 píxeles mal** y cumple el criterio de ≤ 1 píxel a partir de la
época 450.

### Espacio latente

El notebook grafica los 32 códigos en el plano y el "atlas" del decodificador, que decodifica
una grilla regular del plano. Los códigos quedan entre −4,23 y 7,5 y los dos más cercanos
están a 0,39 de distancia. Las letras parecidas tienden a quedar cerca, aunque la pérdida
no lo exige: es un efecto de que el decodificador sea una función suave.

### Generación de una letra nueva

El decodificador está definido en todo el plano, así que decodificar un punto que no es
el código de ninguna letra produce una imagen nueva. Se usan dos estrategias:

- **Interpolación** entre los códigos de "o" y "c". Los puntos intermedios generan imágenes
  a una distancia de Hamming de 2 a 6 píxeles de todo el conjunto de entrenamiento.
- **Barrido de una grilla** de 41×41 puntos dentro del rango de los códigos. 528 de 1681
  puntos generan imágenes a Hamming ≥ 2 de todas las letras, y 171 de ellas además son
  nítidas (a lo sumo 2 píxeles grises).

Entre esas candidatas se elige a ojo la más legible, un trazo entre "b" y "k", y se muestra
junto a sus dos vecinos más próximos: está a 3 píxeles de la "b" y a 7 de la "r".

## Ejercicio 1b: denoising autoencoder

Se entrena con la entrada corrupta y se exige reconstruir la entrada limpia (Vincent et
al., 2008). Así la red tiene que aprender qué combinaciones de píxeles forman letras.

| Decisión | Elección | Justificación |
|---|---|---|
| Ruido | Cada píxel se invierte con probabilidad p | Es el ruido natural para imágenes binarias y p se interpreta como fracción de píxeles dañados |
| Resampleo | Ruido nuevo en cada época | Con sólo 32 patrones, un ruido fijo se memorizaría |
| Arquitectura | 35-32-16-d-16-32-35, tanh, Adam (lr = 0,005), 4000 épocas | La misma del ejercicio 1a, para aislar el efecto del ruido |
| Latente | d = 2 y d = 8 | d = 8 permite comparar con un cuello de botella menos estrecho |
| Ruido de entrenamiento | p ∈ {0; 0,05; 0,1; 0,2} | Más ruido enseña robustez pero cuesta precisión sin ruido |

**Evaluación.** Para cada nivel de ruido de test se mide el promedio de píxeles errados
de la reconstrucción respecto del patrón limpio (32 letras × 20 muestras de ruido × 5
semillas, con semillas de ruido distintas de las de entrenamiento).

Píxeles errados tras reconstruir (de 35), según el ruido de test:

| d | p de entrenamiento | 0 | 0,05 | 0,1 | 0,2 | 0,3 |
|---|---|---|---|---|---|---|
| – | sin red (entrada) | 0 | 1,73 | 3,48 | 7,02 | 10,54 |
| 2 | 0 | 0,00 | 5,44 | 8,21 | 11,01 | 13,26 |
| 2 | 0,05 | 0,05 | 0,21 | 1,19 | 5,32 | 10,09 |
| 2 | 0,1 | 0,20 | 0,30 | 0,93 | 4,44 | 9,11 |
| 2 | 0,2 | 1,81 | 1,87 | 2,19 | 4,18 | 8,13 |
| 8 | 0 | 0,00 | 2,18 | 4,55 | 8,66 | 12,19 |
| 8 | 0,05 | 0,00 | 0,09 | 0,67 | 3,96 | 8,48 |
| 8 | 0,1 | 0,02 | 0,08 | 0,44 | 2,87 | 7,53 |
| 8 | 0,2 | 0,09 | 0,19 | 0,43 | 2,34 | 6,64 |

- **Un autoencoder entrenado sin ruido empeora la entrada.** Con 2,5 % de píxeles
  invertidos (0,9 píxeles en la entrada) devuelve 3,3 píxeles errados con d = 2 y 1,1 con
  d = 8.
- **El denoising autoencoder sí limpia.** Entrenado con p = 0,1 y d = 8, con ruido de test
  0,1 (3,5 píxeles errados en la entrada) deja 0,44, un 87 % menos. Con d = 2 el resultado
  es peor (0,93), porque el cuello de botella conserva menos información.
- **Hay un compromiso sesgo-varianza.** Entrenar con más ruido protege frente a ruidos altos
  pero cuesta precisión sin ruido: con d = 2 y p = 0,2 aun la entrada limpia se reconstruye
  con 1,8 píxeles errados. Con d = 8 ese costo casi desaparece (0,09).
- **Más ruido que el de entrenamiento degrada de forma gradual.** Cerca de p = 0,3 la
  entrada queda más cerca de otra letra y la red la "corrige" hacia la letra equivocada.
  Es el equivalente de los estados espurios de Hopfield (TP4).
- **Elección.** d = 8 y p de entrenamiento entre 0,1 y 0,2. Con d = 2 conviene un p
  moderado (0,05 a 0,1).

## Conclusiones del ejercicio 1

1. Es posible representar las 32 letras en dos dimensiones con a lo sumo 1 píxel de error
   (el modelo final no comete ninguno), pero no cualquier configuración lo logra: hace
   falta un decodificador de dos capas ocultas, y Adam es el único optimizador que lo
   consigue en todas las semillas de esas arquitecturas.
2. El espacio latente no es interpretable por construcción: las vecindades salen de la
   suavidad del decodificador, no de un objetivo explícito. Esto motiva el VAE del
   ejercicio 2.
3. Una letra nueva sólo es legible cerca de los códigos de entrenamiento. Lejos de ellos
   el decodificador no fue entrenado y produce imágenes degeneradas.
4. El denoising autoencoder elimina buena parte del ruido hasta un umbral, con el costo de
   correcciones equivocadas y seguras más allá de él.

**Limitaciones.** Sólo hay 32 patrones y no hay conjunto de validación: el objetivo es
representar el conjunto, no generalizar. El criterio de "parece una letra" es subjetivo.
Se evaluó un único tipo de ruido (inversión de píxeles).

## Estructura

```text
TP5/
├── ejercicio1.ipynb
├── ejercicio2.ipynb
├── ejercicio2_propuesta_*.ipynb   #Propuestas alternativas de datos para el ejercicio 2
├── deep.py                        #MLP, SGD/Momentum/Adam, Autoencoder y VAE en numpy
├── datos.py                       #Parser de font.h, dataset de símbolos y funciones de ruido
├── experimentos.py                #Entrenamiento y barridos (ejercicios 1 y 2)
├── graficos.py                    #Funciones de gráfico de los notebooks
├── correr_ej2.py                  #Corre el barrido del ejercicio 2 para un tamaño de latente
├── config.json                    #Hiperparámetros y semillas
├── SIATP5.pdf                     #Enunciado
├── data/
│   └── font.h                     #32 caracteres de 7×5 (0x60 a 0x7f)
└── results/
    ├── ej1a_barrido.json          #Barrido de arquitectura y optimizador
    └── ej1b_barrido.json          #Barrido del denoising autoencoder
```

## Bibliografía

- Goodfellow, Bengio & Courville (2016), *Deep Learning*, cap. 14 y §8.4.
- Aggarwal (2023), *Neural Networks and Deep Learning*, §3.4 y §5.10.2.
- Vincent, Larochelle, Bengio & Manzagol (2008), *Extracting and composing robust features
  with denoising autoencoders*, ICML.
- Hinton & Salakhutdinov (2006), *Reducing the dimensionality of data with neural
  networks*, Science 313.
- Baldi & Hornik (1989), *Neural networks and principal component analysis*, Neural
  Networks 2.
- Alain & Bengio (2014), *What regularized auto-encoders learn from the data-generating
  distribution*, JMLR 15.
