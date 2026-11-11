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
| `ejercicio2_pokemon.ipynb` | VAE sobre los 809 sprites de `data/pokemon/` |
| `implementations/ejercicio2.ipynb` | Implementación anterior del ejercicio 2: VAE sobre símbolos matemáticos |

Todos los hiperparámetros y semillas están en `config.json`. Los barridos del ejercicio 1
ya calculados están en `results/`. Si esos archivos existen, el notebook los lee y corre en
unos 15 segundos. Si se borran, los recalcula (unas 130 redes, varios minutos).

Los notebooks del ejercicio 2 también leen sus barridos de `results/`:

- `ejercicio2_pokemon.ipynb` necesita `results/ej2p_barrido.json` y los modelos
  `results/ej2p_*.npz`, que genera `python correr_ej2_pokemon.py 8` (24 redes en 8 procesos,
  unos 25 minutos). Con esos archivos el notebook corre en unos 15 segundos. Sin el barrido
  lo recalcula en serie, y sin los modelos los reentrena.
- `implementations/ejercicio2.ipynb` necesita `results/ej2_beta_d{2,8,16}.json`, que se
  generan con `python correr_ej2.py 2`, `8` y `16` desde `TP5/`. El notebook busca los
  módulos y `results/` en `TP5/`, así que funciona con el kernel en `TP5/` o en
  `TP5/implementations/`.

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

## Ejercicio 2: autoencoder variacional (VAE)

**Objetivo del enunciado:** extender el autoencoder anterior a un VAE sobre un conjunto de
datos nuevo, para resolver el problema de la representación en el espacio latente, y usarlo
para generar una muestra que se pueda juzgar como parte del conjunto.

**Modelo.** `deep.VAE` reusa las piezas del autoencoder del ejercicio 1 (capas `MLP`,
retropropagación a mano, Adam y BCE desde logits). El codificador devuelve μ y log σ² en
lugar de un punto. El código se muestrea con el truco de reparametrización,
z = μ + σ·ε, y la pérdida es BCE + β·KL(q(z|x) ‖ N(0, I)). Con β = 1 es el ELBO exacto y con
β = 0 es un autoencoder determinístico, que sirve de comparación directa con el ejercicio 1.
Los gradientes se verifican con diferencias finitas (error relativo máximo 1,8·10⁻⁸).

**Cómo se mide si una muestra "pertenece".** Se usa la distancia de Fréchet (la idea del
FID) entre las muestras del prior y el conjunto real, comparada con un piso: la misma
distancia para datos reales que el modelo no vio. Además se mide la fracción de muestras
que son casi copias de un dato de entrenamiento.

El ejercicio se resolvió con dos conjuntos de datos y el mismo modelo. La implementación
anterior, con símbolos matemáticos, está en `implementations/`: ahí el VAE funciona bien. La
actual, con Pokémon, muestra hasta dónde llega una MLP escrita en numpy.

### Implementación anterior: símbolos matemáticos (`implementations/ejercicio2.ipynb`)

**Datos.** 39 símbolos (+, ×, ≤, ∑, ∫, ∂, √, π…) × 8 tipografías DejaVu × 9 desplazamientos
de ±1 píxel: 2808 imágenes de 16×16, generadas desde las fuentes que trae `matplotlib`. Se
retienen dos tipografías completas como validación. Un clasificador propio (99,86 % de
exactitud en tipografías no vistas) hace de juez, y la distancia de Fréchet se calcula sobre
sus rasgos internos.

**Modelo.** 256-128-64-d, tanh, Adam (lr = 0,002), lotes de 128, 150 épocas. Barrido de
β ∈ {0; 0,25; 1; 4} y d ∈ {2; 8; 16}, con 3 semillas por combinación.

| d | β | KL (nats) | Píxeles errados (de 256), tipografías no vistas | Fréchet del prior |
|---|---|---|---|---|
| 2 | 0 | 29,1 | 30,3 | 156,4 |
| 2 | 1 | 5,6 | 29,3 | 33,6 |
| 16 | 0 | 104,0 | 9,0 | 358,6 |
| 16 | 0,25 | 28,1 | 9,2 | 14,3 |
| 16 | 1 | 13,5 | 12,5 | 13,5 |
| 16 | 4 | 3,7 | 28,0 | 40,1 |

El piso de la distancia de Fréchet es 14,2.

- **El KL vuelve muestreable el espacio latente.** Con d = 16, las muestras del VAE (β = 1)
  quedan al nivel del piso. El autoencoder con la misma arquitectura queda unas 27 veces más
  lejos.
- **El costo es la reconstrucción:** 12,5 píxeles errados contra 9,0 del autoencoder.
  β = 4 sobrerregulariza: reconstruye peor y genera peor.
- **Generación.** De 1500 muestras del prior del VAE con d = 16, 382 son reconocidas por el
  juez con confianza ≥ 0,9 y no son copias. Se muestran ocho, cada una junto a su vecino real
  más cercano.
- **La métrica importa.** La confianza del juez favorece al autoencoder (0,47 contra 0,27
  de muestras reconocibles), porque premia imágenes sobreconfiadas. La distancia de Fréchet
  contra un piso real es la que separa los modelos.

### Sprites de Pokémon (`ejercicio2_pokemon.ipynb`)

**Datos.** Los 809 sprites de `data/pokemon/` (generaciones 1 a 7). Cada uno se compone
sobre fondo blanco, se recorta al Pokémon y se escala a 32×32 RGB (3072 valores). Se
retienen 81 Pokémon completos como validación. Los 728 de entrenamiento se duplican con su
reflejo horizontal. No hay clases ni juez: la distancia de Fréchet se calcula sobre las 20
primeras componentes principales del entrenamiento (SVD de numpy).

**Modelo.** 3072-512-256-d, ReLU, Adam (lr = 3·10⁻⁴), lotes de 64, 300 épocas. Con tanh las
3072 entradas saturan la primera capa, y con β = 1 el modelo colapsaba: el KL caía a cero y
el decodificador devolvía siempre la imagen media. Barrido de β ∈ {0; 0,25; 1; 4} y
d ∈ {2; 32}, con 3 semillas por combinación.

| d | β | KL (nats) | BCE de reconstrucción, Pokémon no vistos | Fréchet del prior | Casi copias |
|---|---|---|---|---|---|
| 2 | 0 | 14,8 | 896 | 102,5 | 2 % |
| 2 | 1 | 6,4 | 895 | 52,6 | 8 % |
| 32 | 0 | 298,7 | 802 | 19,1 | 0 % |
| 32 | 0,25 | 74,9 | 818 | 14,5 | 0 % |
| 32 | 1 | 37,1 | 815 | 6,8 | 1 % |
| 32 | 4 | 11,0 | 835 | 7,4 | 9 % |

La BCE va de 600 nats (reconstrucción perfecta) a 981 (predecir siempre la imagen media). El
piso de la distancia de Fréchet es 4,1.

- **El KL vuelve muestreable el espacio latente.** Con d = 32, el autoencoder (β = 0)
  dispersa sus códigos (KL de 299 nats). Con z ~ N(0, I) su decodificador produce ruido de
  colores, porque esa zona nunca la vio: es el problema de los huecos del ejercicio 1. El
  VAE genera figuras centradas sobre fondo blanco, con Fréchet 6,8 frente al piso de 4,1.
- **β = 1 es el equilibrio.** β = 4 tiene mejor ELBO de validación (849 contra 862 nats),
  pero no mejora las muestras y multiplica por 9 las casi copias.
- **d = 2 sirve para ver, no para dibujar.** El plano se ordena solo por color y silueta,
  pero reconstruye cerca de la imagen media (≈ 895 nats). Dos números alcanzaban para las 32
  letras del ejercicio 1, no para 728 Pokémon distintos.
- **Generación.** De 2000 muestras del prior, 629 no son copias y tienen el área y el
  contraste de un sprite real. Su distancia mediana al conjunto en el espacio PCA es la de un
  Pokémon real no visto (4,56 contra 4,54). La muestra elegida es una criatura bípeda rojiza
  cuyos vecinos más cercanos son Charmander, Dragonite y Charmeleon, y no es ninguno de ellos.
- **Las interpolaciones no distinguen los modelos.** Entre dos códigos de entrenamiento el
  autoencoder también interpola bien. La diferencia aparece al muestrear del prior.

**Limitación principal.** Las muestras son borrosas y de colores apagados: se reconoce la
silueta y la paleta, pero no ojos ni contornos. Es el límite de una MLP con verosimilitud
factorizada por píxel y 728 ejemplos. Para mejorarlo haría falta una red convolucional.

## Conclusiones del ejercicio 2

1. El término KL resuelve el problema del ejercicio 1. En los dos conjuntos, el mismo modelo
   sin KL genera imágenes inservibles desde N(0, I), y con KL genera muestras cercanas al piso
   de datos reales: Fréchet 13,5 contra 14,2 en símbolos y 6,8 contra 4,1 en Pokémon.
2. El precio es reconstruir algo peor. β = 1 es el mejor equilibrio en los dos casos.
3. Con símbolos el VAE genera muestras nítidas que un clasificador independiente reconoce.
   Con Pokémon genera criaturas plausibles pero borrosas: el límite ya no es el esquema
   variacional sino la capacidad de una MLP frente a datos muy diversos y en color.
4. Interpolar entre datos conocidos no prueba que un modelo sea generativo. Muestrear del
   prior sí.

**Limitaciones.** Hay 3 semillas por combinación y una sola partición de validación. En
Pokémon, la distancia de Fréchet sobre 20 componentes principales es un sustituto grueso
del FID, y el filtro que elige la muestra nueva es una heurística. Los dos conjuntos son de
imágenes sintéticas o dibujadas, no manuscritas.

## Estructura

```text
TP5/
├── ejercicio1.ipynb
├── ejercicio2_pokemon.ipynb
├── deep.py                        #MLP, SGD/Momentum/Adam, Autoencoder y VAE en numpy
├── datos.py                       #Parser de font.h, dataset de símbolos y funciones de ruido
├── experimentos.py                #Entrenamiento y barridos (ejercicios 1 y 2)
├── graficos.py                    #Funciones de gráfico de los notebooks
├── correr_ej2.py                  #Barrido del ejercicio 2 con símbolos para un tamaño de latente
├── correr_ej2_pokemon.py          #Barrido β × latente del VAE de Pokémon, en paralelo
├── config.json                    #Hiperparámetros y semillas
├── SIATP5.pdf                     #Enunciado
├── implementations/
│   └── ejercicio2.ipynb           #Implementación anterior del ejercicio 2 (símbolos matemáticos)
├── data/
│   ├── font.h                     #32 caracteres de 7×5 (0x60 a 0x7f)
│   └── pokemon/                   #809 sprites de 120×120 (PNG y JPG)
└── results/
    ├── ej1a_barrido.json          #Barrido de arquitectura y optimizador
    ├── ej1b_barrido.json          #Barrido del denoising autoencoder
    ├── ej2p_barrido.json          #Barrido β × latente del VAE de Pokémon
    └── ej2p_b*_d*_s0.npz          #Modelos finales de Pokémon (β = 0 y 1, d = 2 y 32)
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
- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*, ICLR.
- Doersch (2016), *Tutorial on Variational Autoencoders*, arXiv:1606.05908.
- Higgins et al. (2017), *β-VAE: Learning basic visual concepts with a constrained
  variational framework*, ICLR.
- Bowman et al. (2016), *Generating sentences from a continuous space*, CoNLL.
- Heusel et al. (2017), *GANs trained by a two time-scale update rule converge to a local
  Nash equilibrium*, NeurIPS.
