# TP1 - Métodos de búsqueda para Sokoban

El programa resuelve tableros de Sokoban con métodos de búsqueda desinformados (BFS, DFS e
IDDFS) e informados (Greedy y A*), y compara su calidad de solución y su costo
computacional sobre 101 mapas. Los algoritmos, las heurísticas y la detección de deadlocks
son propios; pandas y matplotlib se usan únicamente para registrar y graficar los
resultados.

## Ejecución

Desde la raíz del repositorio:

```bash
uv sync
uv run --with jupyter jupyter lab TP1/TP1_Sokoban_78.ipynb
```

También se puede abrir `TP1/TP1_Sokoban_78.ipynb` en VS Code y seleccionar
`.venv/bin/python` como kernel. El notebook busca `mapas/` y `output/` tanto si se abre
desde `TP1/` como desde la raíz.

El notebook carga un mapa personalizado definido en `mapa_raw` y todos los `.txt` de
`TP1/mapas/`. Cada mapa usa la notación estándar: `#` pared, `@` jugador, `$` caja,
`.` objetivo.

## Modelado

El estado (`SokobanState`) guarda la posición del jugador y las posiciones de las cajas,
ordenadas para que el hash sea único. Paredes y objetivos son estáticos, por lo que no
participan de la igualdad entre estados. Las acciones son `U`, `D`, `L` y `R`, cada
movimiento cuesta 1 y el estado es solución cuando todas las cajas están en objetivos.

## Métodos y heurísticas

- Desinformados: BFS, DFS e IDDFS (profundidad máxima 100).
- Informados: Greedy y A*, cada uno con las tres heurísticas.
- `h_manhattan`: suma, para cada caja, la distancia Manhattan al objetivo más cercano.
  Admisible y consistente.
- `h_manhattan_deadlock`: Manhattan más la distancia del jugador a la caja más cercana,
  e infinito si se detecta un deadlock. Admisible y domina a la Manhattan.
- `h_no_admisible`: 2 × Manhattan. Se incluye para mostrar el efecto de perder la
  admisibilidad.

Todas las búsquedas se cortan a los 200.000 nodos expandidos. En total se ejecutan 909
búsquedas (101 mapas × 9 configuraciones).

## Estructura

```text
TP1/
├── TP1_Sokoban_78.ipynb        #Estado, heurísticas, algoritmos, experimentos y gráficos
├── ITBA_TP1_IA.pdf             #Presentación
├── mapas/                      #100 mapas estándar (mapa_000.txt a mapa_099.txt)
├── output/                     #Resultados generados por el notebook
└── deprecated/                 #Implementación anterior
```

## Resultados

La última parte del notebook crea `TP1/output/` con:

- `resultados_experimentos.csv`     #Una fila por mapa y configuración
- `graficos_comparativos.png`
- `costo_soluciones.png`
- `boxplot_nodos_por_metodo.png`
- `boxplot_tiempo_por_metodo.png`
- `distribucion_nodos_por_metodo.png`
- `distribucion_tiempo_por_metodo.png`
- `scatter_tiempo_nodos_top3.png`
- `scatter_tiempo_costo_top3.png`
- `scatter_nodos_frontera_top3.png`
- `solucion_A_estrella.gif`         #Solución del mapa personalizado con A* + deadlocks

| Método | Éxito | Costo mediano | Nodos medianos | Tiempo mediano |
|---|---|---|---|---|
| A* + deadlocks | 100 % | 19 | 495 | 0,0082 s |
| BFS | 100 % | 19 | 1947 | 0,0123 s |
| Greedy + deadlocks | 100 % | 26 | 84 | 0,0014 s |
| DFS | 100 % | 48 | 803 | 0,0046 s |
| IDDFS | 44,5 % | 16* | 200.000 | 0,6632 s |

\* Calculado sólo sobre los 45 mapas que IDDFS resolvió.

A* con `h_manhattan_deadlock` obtiene el mismo costo que BFS expandiendo 74,6 % menos
nodos, y la detección de deadlocks reduce 53,2 % los nodos de A* frente a la Manhattan
simple. Greedy es la opción más rápida, con soluciones 37 % más largas.
