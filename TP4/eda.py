"""Resúmenes tabulares para el análisis exploratorio."""

import numpy as np
import pandas as pd


def describe(X: pd.DataFrame) -> pd.DataFrame:
    """Estadísticas descriptivas con asimetría y curtosis en exceso."""
    summary = X.describe().T[["mean", "std", "min", "50%", "max"]]
    summary["cv"] = summary["std"] / summary["mean"].abs()
    summary["asimetría"] = X.skew()
    summary["curtosis"] = X.kurt()
    return summary.rename(columns={"mean": "media", "std": "desvío", "50%": "mediana"})


def correlated_pairs(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Pares de variables con |r de Pearson| por encima del umbral."""
    corr = X.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strong = upper[upper.abs() > threshold].sort_values(key=np.abs, ascending=False)
    return strong.rename("r").rename_axis(["variable 1", "variable 2"]).reset_index()
