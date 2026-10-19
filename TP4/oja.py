"""Regla de Oja: una neurona lineal que aprende la primera componente principal."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class Oja:
    """Neurona lineal y = wᵀx entrenada online con Δw = η·y·(x − y·w).

    El término −η·y²·w normaliza los pesos sin hacerlo explícitamente: w converge a un
    vector de norma 1 en la dirección de máxima varianza de los datos (centrados).
    """

    learning_rate: float = 1e-3
    epochs: int = 200
    seed: int = 0
    weights_: np.ndarray = field(init=False, repr=False)
    history_: np.ndarray = field(init=False, repr=False)
    diverged_: bool = field(init=False, default=False)

    def fit(self, X: np.ndarray) -> "Oja":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        w = rng.uniform(-1, 1, X.shape[1])
        w /= np.linalg.norm(w)
        history = [w.copy()]
        self.diverged_ = False

        for _ in range(self.epochs):
            # Con tasas grandes los pesos explotan: se detecta al final de la época.
            with np.errstate(over="ignore", invalid="ignore"):
                for index in rng.permutation(len(X)):
                    x = X[index]
                    y = w @ x
                    w = w + self.learning_rate * y * (x - y * w)
                exploded = not np.all(np.isfinite(w)) or np.linalg.norm(w) > 1e6
            if exploded:
                self.diverged_ = True
                break
            history.append(w.copy())

        self.weights_ = w
        self.history_ = np.array(history)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Índice de cada muestra: su proyección sobre los pesos aprendidos."""
        return np.asarray(X, dtype=float) @ self.weights_


def orient(w: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Devuelve w con el signo que la alinea con `reference` (el signo de Oja es arbitrario)."""
    return w if w @ reference >= 0 else -w


def alignment(history: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """|cos| entre cada vector de la historia y una dirección de referencia."""
    norms = np.linalg.norm(history, axis=1) * np.linalg.norm(direction)
    return np.abs(history @ direction) / norms
