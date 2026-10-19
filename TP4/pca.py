"""Análisis de Componentes Principales implementado con numpy."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(slots=True)
class PCA:
    """PCA por descomposición espectral de la matriz de covarianza.

    Si los datos están estandarizados, esa matriz es la de correlación y cada
    autovalor se mide en "variables equivalentes" (suman la cantidad de variables).
    """

    n_components: int | None = None
    mean_: np.ndarray = field(init=False)
    eigenvalues_: np.ndarray = field(init=False)
    eigenvectors_: np.ndarray = field(init=False)

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        covariance = centered.T @ centered / len(X)

        # eigh aprovecha la simetría y devuelve autovalores ascendentes.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)  # Quita el ruido numérico negativo.
        eigenvectors = eigenvectors[:, order]

        # El signo de un autovector es arbitrario: se fija para que su coeficiente
        # de mayor magnitud sea positivo y el resultado sea reproducible.
        largest = np.argmax(np.abs(eigenvectors), axis=0)
        eigenvectors *= np.sign(eigenvectors[largest, np.arange(eigenvectors.shape[1])])

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        return self

    @property
    def k(self) -> int:
        return self.n_components or len(self.eigenvalues_)

    @property
    def components_(self) -> np.ndarray:
        """Autovectores retenidos, uno por fila: (k, n_variables)."""
        return self.eigenvectors_[:, : self.k].T

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.eigenvalues_ / self.eigenvalues_.sum()

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        return scores @ self.components_ + self.mean_

    def loadings(self, features: list[str]) -> pd.DataFrame:
        """Correlación entre cada variable y cada componente (datos estandarizados)."""
        values = self.components_.T * np.sqrt(self.eigenvalues_[: self.k])
        return pd.DataFrame(values, index=features, columns=[f"PC{i + 1}" for i in range(self.k)])


def broken_stick(n: int) -> np.ndarray:
    """Proporción de varianza esperada por componente si la varianza se repartiera al azar."""
    return np.array([np.sum(1.0 / np.arange(i, n + 1)) / n for i in range(1, n + 1)])


def retention_criteria(eigenvalues: np.ndarray, thresholds=(0.80, 0.90)) -> pd.Series:
    """Cantidad de componentes que retiene cada criterio clásico."""
    ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ratio)
    beats_stick = ratio > broken_stick(len(ratio))
    criteria = {
        "Kaiser (λ > 1)": int(np.sum(eigenvalues > 1)),
        # Se cuentan las primeras componentes consecutivas que superan al bastón roto.
        "Bastón roto": int(np.argmin(beats_stick)) if not beats_stick.all() else len(ratio),
    }
    for threshold in thresholds:
        criteria[f"Varianza acumulada ≥ {threshold:.0%}"] = int(np.argmax(cumulative >= threshold) + 1)
    return pd.Series(criteria, name="componentes")

