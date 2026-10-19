"""Carga, selección de variables y estandarización del catálogo de creadores.

Los ejercicios 1.2 (Kohonen) y 1.3 (Oja) usan `prepare_features` para trabajar con
exactamente las mismas variables y la misma transformación que el PCA del 1.1.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ["creator_id", "name"]

# Columnas que el enunciado define como función exacta de otras.
DERIVED_COLUMNS = {
    "price_ars": "price_usd · 1500",
    "estimated_argentina_reach": "average_reach · argentina_audience_pct / 100",
    "total_engagement_per_1000": "likes + comments + saves + shares (por 1000)",
    "organic_posts_pct": "100 − sponsored_posts_pct",
}

# Conteos y precios: positivos, abarcan dos órdenes de magnitud y crecen de forma
# multiplicativa, por eso se comparan en escala logarítmica.
SCALE_COLUMNS = [
    "followers",
    "average_reach",
    "average_impressions",
    "average_video_views",
    "price_usd",
]

TOPIC_COLUMNS = ["topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct"]
CONTENT_COLUMNS = [
    "content_tutorial_pct",
    "content_review_pct",
    "content_entertainment_pct",
    "content_showcase_pct",
]

# Grupos temáticos de variables, en el orden en que se muestran en los gráficos.
FEATURE_GROUPS = {
    "Alcance y precio": SCALE_COLUMNS + ["monthly_posts", "non_follower_reach_pct"],
    "Audiencia": ["argentina_audience_pct"],
    "Interacción": [
        "likes_per_1000",
        "comments_per_1000",
        "saves_per_1000",
        "shares_per_1000",
        "profile_visits_per_1000",
    ],
    "Comercial": ["sponsored_posts_pct"],
    "Temática": TOPIC_COLUMNS,
    "Tipo de contenido": CONTENT_COLUMNS,
    "Formato": [
        "average_caption_words",
        "average_watch_time_seconds",
        "video_posts_pct",
        "short_form_posts_pct",
        "trend_posts_pct",
    ],
}

TOPIC_LABELS = {
    "topic_football_pct": "Fútbol",
    "topic_makeup_pct": "Maquillaje",
    "topic_dogs_pct": "Perros",
    "topic_art_pct": "Arte",
}


def find_data_path(filename: str = "creadores.csv") -> Path:
    """Busca el CSV tanto si el notebook se abre desde TP4 como desde la raíz."""
    cwd = Path.cwd()
    for candidate in (cwd / "data" / filename, cwd / "TP4" / "data" / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontró data/{filename} ni TP4/data/{filename}")


def load_creators(path: Path | str | None = None) -> pd.DataFrame:
    return pd.read_csv(path if path is not None else find_data_path())


def check_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Error absoluto máximo entre cada columna derivada y su fórmula."""
    engagement = df[["likes_per_1000", "comments_per_1000", "saves_per_1000", "shares_per_1000"]]
    recomputed = {
        "price_ars": df["price_usd"] * 1500,
        "estimated_argentina_reach": df["average_reach"] * df["argentina_audience_pct"] / 100,
        "total_engagement_per_1000": engagement.sum(axis=1),
        "organic_posts_pct": 100 - df["sponsored_posts_pct"],
    }
    rows = []
    for column, expected in recomputed.items():
        error = (df[column] - expected).abs()
        rows.append({
            "columna": column,
            "fórmula": DERIVED_COLUMNS[column],
            "error máx.": error.max(),
            "error relativo máx.": (error / df[column].abs()).max(),
        })
    return pd.DataFrame(rows).set_index("columna")


def composition_sums(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen de la suma por fila de cada grupo de porcentajes que debería dar 100."""
    groups = {
        "topic_*": TOPIC_COLUMNS,
        "content_*": CONTENT_COLUMNS,
        "sponsored + organic": ["sponsored_posts_pct", "organic_posts_pct"],
    }
    return pd.DataFrame({
        name: df[columns].sum(axis=1).agg(["min", "max"]) for name, columns in groups.items()
    }).T


def select_features(df: pd.DataFrame) -> list[str]:
    """Todas las variables numéricas salvo identificadores y columnas derivadas."""
    excluded = set(ID_COLUMNS) | set(DERIVED_COLUMNS)
    ordered = [column for columns in FEATURE_GROUPS.values() for column in columns]
    missing = set(df.columns) - excluded - set(ordered)
    if missing:
        raise ValueError(f"Columnas sin grupo asignado: {sorted(missing)}")
    return ordered


def dominant_topic(df: pd.DataFrame) -> pd.Series:
    """Temática con mayor porcentaje de publicaciones de cada creador."""
    return df[TOPIC_COLUMNS].idxmax(axis=1).map(TOPIC_LABELS).rename("temática")


def log_transform(X: pd.DataFrame, columns: list[str] = SCALE_COLUMNS) -> pd.DataFrame:
    if (X[columns] <= 0).any().any():
        raise ValueError("La transformación logarítmica requiere valores positivos")
    return X.assign(**{column: np.log10(X[column]) for column in columns})


@dataclass(slots=True)
class Standardizer:
    """Centra y escala cada variable a desvío 1 (desvío poblacional, ddof=0)."""

    mean_: np.ndarray = field(init=False)
    std_: np.ndarray = field(init=False)

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        if np.any(self.std_ == 0):
            raise ValueError("Hay variables constantes; no se pueden estandarizar")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) * self.std_ + self.mean_


@dataclass(slots=True)
class PreparedData:
    """Matriz lista para los modelos junto con lo necesario para interpretarla."""

    X: np.ndarray
    features: list[str]
    standardizer: Standardizer
    transformed: pd.DataFrame
    names: pd.Series
    topic: pd.Series


def prepare_features(df: pd.DataFrame) -> PreparedData:
    """Selección de variables -> log10 en las de escala -> estandarización."""
    features = select_features(df)
    transformed = log_transform(df[features])
    standardizer = Standardizer()
    X = standardizer.fit_transform(transformed.to_numpy())
    return PreparedData(
        X=X,
        features=features,
        standardizer=standardizer,
        transformed=transformed,
        names=df["name"],
        topic=dominant_topic(df),
    )
