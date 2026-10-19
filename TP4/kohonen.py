"""Red de Kohonen (mapa autoorganizado) sobre una grilla rectangular."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pca import PCA


@dataclass(slots=True)
class Kohonen:
    """Mapa autoorganizado entrenado de forma online, una muestra por vez.

    La tasa de aprendizaje y el radio de vecindad decaen exponencialmente durante el
    entrenamiento: la tasa de `learning_rate` a `final_learning_rate` y el radio de
    `radius` a `final_radius`. Un radio final chico deja que cada neurona se ajuste a sus
    propios creadores; uno grande mantiene el mapa más suave.
    """

    rows: int
    cols: int
    learning_rate: float = 0.5
    radius: float | None = None
    epochs: int = 100
    init: str = "sample"
    seed: int = 0
    final_learning_rate: float = 0.01
    final_radius: float = 1.0
    weights_: np.ndarray = field(init=False, repr=False)
    positions_: np.ndarray = field(init=False, repr=False)
    history_: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.init not in ("sample", "pca"):
            raise ValueError("init debe ser 'sample' o 'pca'")
        if self.radius is None:
            self.radius = max(self.rows, self.cols) / 2
        # Coordenadas (fila, columna) de cada neurona, en el orden de `weights_`.
        self.positions_ = np.array(
            [(r, c) for r in range(self.rows) for c in range(self.cols)], dtype=float
        )

    @property
    def n_neurons(self) -> int:
        return self.rows * self.cols

    def _initial_weights(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self.init == "sample":
            # Cada neurona arranca en un creador elegido al azar (con reposición si
            # hay más neuronas que creadores).
            replace = self.n_neurons > len(X)
            return X[rng.choice(len(X), self.n_neurons, replace=replace)].copy()

        # Inicialización lineal: la grilla se extiende sobre el plano PC1-PC2, con las
        # columnas a lo largo de PC1 y las filas a lo largo de PC2 (±2 desvíos).
        pca = PCA(n_components=2).fit(X)
        spread = 2 * np.sqrt(pca.eigenvalues_[:2])
        col_coord = np.linspace(-1, 1, self.cols) if self.cols > 1 else np.zeros(1)
        row_coord = np.linspace(-1, 1, self.rows) if self.rows > 1 else np.zeros(1)
        coords = np.array([(col_coord[int(c)], row_coord[int(r)]) for r, c in self.positions_])
        return pca.mean_ + (coords * spread) @ pca.components_

    def fit(self, X: np.ndarray) -> "Kohonen":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        self.weights_ = self._initial_weights(X, rng)
        self.history_ = []

        total_steps = self.epochs * len(X)
        step = 0
        for _ in range(self.epochs):
            for index in rng.permutation(len(X)):
                progress = step / total_steps
                eta = self.learning_rate * (self.final_learning_rate / self.learning_rate) ** progress
                radius = self.radius * (self.final_radius / self.radius) ** progress

                x = X[index]
                winner = np.argmin(np.sum((self.weights_ - x) ** 2, axis=1))
                grid_dist2 = np.sum((self.positions_ - self.positions_[winner]) ** 2, axis=1)
                neighborhood = np.exp(-grid_dist2 / (2 * radius**2))
                self.weights_ += eta * neighborhood[:, None] * (x - self.weights_)
                step += 1
            self.history_.append(self.quantization_error(X))
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Distancia euclídea de cada muestra a cada neurona: (n_muestras, n_neuronas)."""
        X = np.asarray(X, dtype=float)
        return np.linalg.norm(X[:, None, :] - self.weights_[None, :, :], axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Índice de la neurona ganadora (BMU) de cada muestra."""
        return np.argmin(self._distances(X), axis=1)

    def quantization_error(self, X: np.ndarray) -> float:
        """Distancia media de cada muestra a su neurona ganadora."""
        return float(self._distances(X).min(axis=1).mean())

    def topographic_error(self, X: np.ndarray) -> float:
        """Fracción de muestras cuyas dos neuronas más cercanas no son vecinas en la grilla.

        Se consideran vecinas las 8 neuronas que rodean a cada una (incluye diagonales).
        """
        order = np.argsort(self._distances(X), axis=1)
        first, second = self.positions_[order[:, 0]], self.positions_[order[:, 1]]
        return float(np.mean(np.abs(first - second).max(axis=1) > 1))

    def hits(self, X: np.ndarray) -> np.ndarray:
        """Cantidad de muestras asignadas a cada neurona, como matriz (filas, columnas)."""
        counts = np.bincount(self.predict(X), minlength=self.n_neurons)
        return counts.reshape(self.rows, self.cols)

    def u_matrix(self) -> np.ndarray:
        """Distancia media de cada neurona a sus vecinas inmediatas (arriba, abajo, izquierda, derecha)."""
        W = self.weights_.reshape(self.rows, self.cols, -1)
        U = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = [
                    W[r + dr, c + dc]
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= r + dr < self.rows and 0 <= c + dc < self.cols
                ]
                U[r, c] = np.mean([np.linalg.norm(W[r, c] - w) for w in neighbors])
        return U



def explained_variance(model: Kohonen, X: np.ndarray) -> float:
    """Fracción de la varianza total que reproducen los prototipos (análogo a la de PCA)."""
    X = np.asarray(X, dtype=float)
    residual = np.mean(model._distances(X).min(axis=1) ** 2)
    return float(1 - residual / np.sum(X.var(axis=0)))


def sweep(X: np.ndarray, configs: list[dict], seeds=range(5), **fixed) -> pd.DataFrame:
    """Entrena una red por configuración y semilla, y devuelve sus métricas."""
    rows = []
    for config in configs:
        params = {**fixed, **config}
        for seed in seeds:
            model = Kohonen(**params, seed=seed).fit(X)
            hits = model.hits(X)
            rows.append({
                **config,
                "seed": seed,
                "QE": model.quantization_error(X),
                "TE": model.topographic_error(X),
                "varianza explicada": explained_variance(model, X),
                "neuronas vacías": float(np.mean(hits == 0)),
            })
    return pd.DataFrame(rows)
