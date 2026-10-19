"""Gráficos del análisis exploratorio y del PCA."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"

# Cuatro categorías en un scatter: subconjunto de la paleta que pasa la validación
# de todos los pares (normal y daltonismo). La forma del marcador refuerza el color.
TOPIC_STYLE = {
    "Fútbol": {"color": "#008300", "marker": "o"},
    "Maquillaje": {"color": "#e87ba4", "marker": "s"},
    "Perros": {"color": "#2a78d6", "marker": "^"},
    "Arte": {"color": "#eda100", "marker": "D"},
}

# Divergente azul <-> rojo con punto medio gris neutro.
DIVERGING = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f2323"]
)


@plt.FuncFormatter
def _compact_formatter(value, _):
    """Abrevia miles y millones en los ejes (250000 -> 250k)."""
    for factor, suffix in ((1e6, "M"), (1e3, "k")):
        if abs(value) >= factor:
            return f"{value / factor:g}{suffix}"
    return f"{value:g}"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.family": "sans-serif",
    })


def _topic_legend(ax, **kwargs) -> None:
    handles = [
        Line2D([], [], linestyle="", markersize=7, markeredgecolor="white",
               markerfacecolor=style["color"], marker=style["marker"], label=topic)
        for topic, style in TOPIC_STYLE.items()
    ]
    ax.legend(handles=handles, title="Temática dominante", title_fontsize=8, **kwargs)


def _scatter_by_topic(ax, x, y, topic: pd.Series, size=42) -> None:
    for name, style in TOPIC_STYLE.items():
        mask = (topic == name).to_numpy()
        ax.scatter(np.asarray(x)[mask], np.asarray(y)[mask], s=size, c=style["color"],
                   marker=style["marker"], edgecolors="white", linewidths=1.0, alpha=0.9, zorder=3)


def _group_separators(ax, groups: dict[str, list[str]], features: list[str]) -> None:
    """Líneas que separan los bloques de variables en un heatmap cuadrado."""
    position = 0
    for columns in groups.values():
        position += sum(column in features for column in columns)
        if position < len(features):
            ax.axhline(position - 0.5, color=SURFACE, linewidth=2)
            ax.axvline(position - 0.5, color=SURFACE, linewidth=2)


def plot_distributions(X: pd.DataFrame, ncols: int = 6) -> plt.Figure:
    nrows = int(np.ceil(X.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.0 * nrows))
    for ax, column in zip(axes.flat, X.columns):
        ax.hist(X[column], bins=15, color=SERIES, edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"{column}\nasimetría {X[column].skew():+.2f}", fontsize=8)
        ax.xaxis.set_major_formatter(_compact_formatter)
        ax.grid(axis="x", visible=False)
    for ax in axes.flat[X.shape[1]:]:
        ax.set_visible(False)
    fig.suptitle("Distribución de las variables originales (50 creadores)", color=INK)
    fig.tight_layout()
    return fig


def plot_correlation(X: pd.DataFrame, groups: dict[str, list[str]]) -> plt.Figure:
    corr = X.corr()
    fig, ax = plt.subplots(figsize=(10, 8.5))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.index)
    ax.grid(False)
    _group_separators(ax, groups, list(corr.columns))
    fig.colorbar(image, ax=ax, shrink=0.75, label="r de Pearson")
    ax.set_title("Matriz de correlación (variables agrupadas por bloque)")
    fig.tight_layout()
    return fig


def plot_annotated_heatmap(table: pd.DataFrame, title: str, colorbar_label: str,
                           limit: float | None = None, figsize=(6, 8)) -> plt.Figure:
    """Heatmap divergente centrado en 0 con el valor escrito en cada celda."""
    limit = limit or float(np.abs(table.to_numpy()).max())
    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(table, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.xaxis.tick_top()
    ax.grid(False)
    for (row, col), value in np.ndenumerate(table.to_numpy()):
        ink = "white" if abs(value) > 0.6 * limit else INK
        ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=7, color=ink)
    fig.colorbar(image, ax=ax, shrink=0.6, label=colorbar_label)
    ax.set_title(title, pad=24)
    fig.tight_layout()
    return fig


def plot_reach_vs_depth(df: pd.DataFrame, topic: pd.Series, depth_columns: list[str]) -> plt.Figure:
    fig, axes = plt.subplots(1, len(depth_columns), figsize=(5.2 * len(depth_columns), 4.2),
                             sharex=True)
    for ax, column in zip(np.atleast_1d(axes), depth_columns):
        _scatter_by_topic(ax, df["followers"], df[column], topic)
        ax.set_xscale("log")
        r = np.corrcoef(np.log10(df["followers"]), df[column])[0, 1]
        ax.set(xlabel="followers (escala log)", ylabel=column,
               title=f"{column} vs. followers  (r con log10 = {r:+.2f})")
    _topic_legend(np.atleast_1d(axes)[-1], loc="upper right")
    fig.tight_layout()
    return fig


def plot_scree(eigenvalues: np.ndarray, stick_ratio: np.ndarray, thresholds=(0.80, 0.90)) -> plt.Figure:
    n = len(eigenvalues)
    positions = np.arange(1, n + 1)
    ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ratio)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    axes[0].bar(positions, eigenvalues, color=SERIES, width=0.7)
    axes[0].axhline(1, color=INK_SECONDARY, linestyle="--", linewidth=1)
    axes[0].text(n, 1.15, "Kaiser: λ = 1", ha="right", fontsize=8, color=INK_SECONDARY)
    axes[0].plot(positions, stick_ratio * eigenvalues.sum(), color="#e34948", linewidth=2,
                 marker="o", markersize=4, label="Bastón roto (azar)")
    axes[0].set(xlabel="Componente", ylabel="Autovalor (variables equivalentes)",
                title="Autovalores de la matriz de correlación")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="x", visible=False)

    axes[1].plot(positions, cumulative, color=SERIES, linewidth=2, marker="o", markersize=4)
    for threshold in thresholds:
        k = int(np.argmax(cumulative >= threshold) + 1)
        axes[1].axhline(threshold, color=INK_MUTED, linestyle="--", linewidth=1)
        axes[1].annotate(f"{threshold:.0%} con {k} componentes", (k, cumulative[k - 1]),
                         xytext=(8, -14), textcoords="offset points", fontsize=8, color=INK_SECONDARY)
    axes[1].set(xlabel="Cantidad de componentes", ylabel="Varianza explicada acumulada",
                ylim=(0, 1.02), title="Varianza explicada acumulada")
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes:
        ax.set_xticks(positions[::2])
    fig.tight_layout()
    return fig


def _merge_close_arrows(load: np.ndarray, indices, tolerance: float) -> list[list[int]]:
    """Agrupa flechas cuyos extremos están a menos de `tolerance` para rotularlas juntas."""
    groups: list[list[int]] = []
    for index in indices:
        for group in groups:
            if np.linalg.norm(load[index] - load[group[0]]) < tolerance:
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def plot_biplot(scores: np.ndarray, loadings: pd.DataFrame, topic: pd.Series, names: pd.Series,
                pcs=(1, 2), explained: np.ndarray | None = None, arrows_per_axis: int = 6,
                n_names: int = 6) -> plt.Figure:
    """Scores coloreados por temática y flechas de las variables con mayor carga en cada eje."""
    i, j = pcs[0] - 1, pcs[1] - 1
    x, y = scores[:, i], scores[:, j]
    load = loadings.iloc[:, [i, j]].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.axvline(0, color=BASELINE, linewidth=1)
    _scatter_by_topic(ax, x, y, topic, size=55)

    # Se eligen las variables de mayor |carga| en cada eje por separado: si se ordenara por
    # norma en el plano, PC1 (mucho más fuerte) se llevaría todas las flechas.
    selected = list(dict.fromkeys(
        list(np.argsort(-np.abs(load[:, 0]))[:arrows_per_axis])
        + list(np.argsort(-np.abs(load[:, 1]))[:arrows_per_axis])
    ))
    # Las cargas son correlaciones (|carga| <= 1): se escalan al rango de los scores.
    scale = 0.9 * min(np.abs(x).max(), np.abs(y).max())
    for group in _merge_close_arrows(load, selected, tolerance=0.2):
        for index in group:
            dx, dy = load[index] * scale
            ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                        arrowprops={"arrowstyle": "-|>", "color": INK_SECONDARY, "linewidth": 1.1})
        dx, dy = load[group].mean(axis=0) * scale
        ax.text(dx * 1.06, dy * 1.06, "\n".join(loadings.index[group]), fontsize=7.5, color=INK,
                ha="left" if dx >= 0 else "right", va="bottom" if dy >= 0 else "top",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8, "pad": 1})

    if n_names > 0:
        order = np.argsort(x)
        for index in np.concatenate([order[: n_names // 2], order[-(n_names // 2):]]):
            ax.annotate(names.iloc[index], (x[index], y[index]), xytext=(5, -10),
                        textcoords="offset points", fontsize=7.5, color=INK_SECONDARY)

    def label(k):
        share = f" ({explained[k]:.1%})" if explained is not None else ""
        return f"PC{k + 1}{share}"

    ax.set(xlabel=label(i), ylabel=label(j),
           title=f"Biplot PC{pcs[0]} vs. PC{pcs[1]}: creadores y variables con mayor carga en cada eje")
    _topic_legend(ax, loc="lower left")
    fig.tight_layout()
    return fig


def plot_scores_by_topic(scores: np.ndarray, topic: pd.Series, n_components: int = 4) -> plt.Figure:
    """Small multiples: posición de cada creador en cada componente, separado por temática."""
    fig, axes = plt.subplots(1, n_components, figsize=(3.4 * n_components, 3.6), sharey=False)
    rng = np.random.default_rng(0)
    topics = list(TOPIC_STYLE)
    for k, ax in enumerate(axes):
        for position, name in enumerate(topics):
            values = scores[(topic == name).to_numpy(), k]
            jitter = rng.uniform(-0.15, 0.15, len(values))
            style = TOPIC_STYLE[name]
            ax.scatter(position + jitter, values, s=30, c=style["color"], marker=style["marker"],
                       edgecolors="white", linewidths=0.8, zorder=3)
            ax.hlines(values.mean(), position - 0.3, position + 0.3, color=INK, linewidth=2, zorder=4)
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_xticks(range(len(topics)), topics, rotation=30)
        ax.set_title(f"PC{k + 1}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score")
    fig.suptitle("Scores por temática dominante (línea negra: media del grupo)", color=INK)
    fig.tight_layout()
    return fig
