"""Gráficos de los ejercicios del TP4."""

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


# Secuencial de un solo tono (azul claro -> oscuro) para magnitudes como la matriz U.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "azul", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"]
)

# Series de líneas en el orden fijo de la paleta categórica.
LINE_SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def plot_som_sweep(results: pd.DataFrame, x: str, hue: str,
                   metrics=("QE", "TE", "varianza explicada", "neuronas vacías")) -> plt.Figure:
    """Media ± desvío entre semillas de cada métrica, una línea por valor de `hue`."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    summary = results.groupby([hue, x])[list(metrics)].agg(["mean", "std"])
    for ax, metric in zip(axes, metrics):
        for color, value in zip(LINE_SERIES, summary.index.get_level_values(0).unique()):
            data = summary.loc[value, metric]
            ax.errorbar(data.index, data["mean"], yerr=data["std"], color=color, linewidth=2,
                        marker="o", markersize=5, capsize=3, label=f"{hue} = {value}")
        ax.set(title=metric, xlabel=x)
        ax.set_xticks(summary.index.get_level_values(1).unique())
        if metric in ("varianza explicada", "neuronas vacías", "TE"):
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def _grid_axes(ax, rows: int, cols: int) -> None:
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set(xlabel="Columna", ylabel="Fila")
    ax.grid(False)


def plot_som_maps(model, X: np.ndarray, topic: pd.Series) -> plt.Figure:
    """Creadores ubicados en su neurona ganadora (izquierda) y matriz U (derecha)."""
    rows, cols = model.rows, model.cols
    winners = model.predict(X)
    hits = model.hits(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    ax = axes[0]
    _grid_axes(ax, rows, cols)
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none",
                                       edgecolor=GRID, linewidth=1))
    for neuron in range(model.n_neurons):
        members = np.flatnonzero(winners == neuron)
        r, c = model.positions_[neuron]
        angles = np.linspace(0, 2 * np.pi, len(members), endpoint=False) + np.pi / 2
        radius = 0.0 if len(members) == 1 else 0.28
        for angle, member in zip(angles, members):
            style = TOPIC_STYLE[topic.iloc[member]]
            ax.scatter(c + radius * np.cos(angle), r - radius * np.sin(angle), s=60,
                       c=style["color"], marker=style["marker"], edgecolors="white",
                       linewidths=1, zorder=3)
    ax.set_title("Creadores por neurona ganadora")
    _topic_legend(ax, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    ax = axes[1]
    U = model.u_matrix()
    image = ax.imshow(U, cmap=SEQUENTIAL)
    _grid_axes(ax, rows, cols)
    for (r, c), value in np.ndenumerate(U):
        ink = "white" if value > U.min() + 0.55 * np.ptp(U) else INK
        label = f"{value:.2f}\n{hits[r, c]} creador{'es' if hits[r, c] != 1 else ''}"
        ax.text(c, r, label, ha="center", va="center", fontsize=7, color=ink)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Distancia media a las neuronas vecinas")
    ax.set_title("Matriz U")
    fig.tight_layout()
    return fig


def plot_planes(planes: np.ndarray, titles: list[str], suptitle: str, colorbar_label: str,
                ncols: int = 4, shared_scale: bool = True) -> plt.Figure:
    """Un heatmap de la grilla por plano (filas, columnas, n_planos).

    Con `shared_scale` todos los planos usan la misma escala y una sola barra de color;
    si no, cada uno usa su propio rango, útil cuando las magnitudes difieren mucho.
    """
    n = planes.shape[2]
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    rows, cols = planes.shape[:2]
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows), squeeze=False)
    for k, (ax, title) in enumerate(zip(axes.flat, titles)):
        limit = float(np.abs(planes if shared_scale else planes[:, :, k]).max())
        image = ax.imshow(planes[:, :, k], cmap=DIVERGING, vmin=-limit, vmax=limit)
        _grid_axes(ax, rows, cols)
        ax.set(title=title, xlabel="", ylabel="")
        if not shared_scale:
            fig.colorbar(image, ax=ax, shrink=0.75)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    if shared_scale:
        fig.colorbar(image, ax=axes, shrink=0.7, label=colorbar_label)
    else:
        fig.supxlabel(colorbar_label, fontsize=9, color=INK_SECONDARY)
    fig.suptitle(suptitle, color=INK)
    return fig


def plot_oja_convergence(histories: dict, direction: np.ndarray) -> plt.Figure:
    """Alineación con PC1 y norma de los pesos por época, una línea por tasa de aprendizaje."""
    colors = LINE_SERIES + ["#eda100"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for color, (label, history) in zip(colors, histories.items()):
        epochs = np.arange(len(history))
        cosine = np.abs(history @ direction) / np.linalg.norm(history, axis=1)
        axes[0].plot(epochs, np.clip(1 - cosine, 1e-8, None), color=color, linewidth=2, label=label)
        axes[1].plot(epochs, np.linalg.norm(history, axis=1), color=color, linewidth=2, label=label)
    axes[0].set(yscale="log", xlabel="Época", ylabel="1 − |cos(w, PC1)|",
                title="Distancia angular a PC1 (escala log)")
    axes[1].axhline(1, color=INK_MUTED, linestyle="--", linewidth=1)
    axes[1].set(xlabel="Época", ylabel="‖w‖", title="Norma de los pesos")
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_oja_weights(weights: np.ndarray, reference: np.ndarray, features: list[str]) -> plt.Figure:
    """Pesos aprendidos por Oja (barras) frente al autovector PC1 del PCA (marcas)."""
    order = np.argsort(weights)
    positions = np.arange(len(features))
    fig, ax = plt.subplots(figsize=(7.5, 8))
    colors = np.where(weights[order] >= 0, "#e34948", SERIES)
    ax.barh(positions, weights[order], color=colors, height=0.7, label="Oja (w)")
    ax.scatter(reference[order], positions, marker="|", s=160, color=INK, linewidths=2,
               zorder=3, label="PCA (autovector PC1)")
    ax.set_yticks(positions, [features[i] for i in order])
    ax.axvline(0, color=BASELINE, linewidth=1)
    ax.set(xlabel="Peso", title="Pesos del índice: Oja frente a PC1")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_index_comparison(index: np.ndarray, pc1_scores: np.ndarray, followers: pd.Series,
                          topic: pd.Series) -> plt.Figure:
    """Índice de Oja contra la proyección sobre PC1 y contra los seguidores."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _scatter_by_topic(axes[0], pc1_scores, index, topic)
    limits = [min(index.min(), pc1_scores.min()), max(index.max(), pc1_scores.max())]
    axes[0].plot(limits, limits, color=INK_MUTED, linestyle="--", linewidth=1, zorder=1)
    axes[0].set(xlabel="Proyección sobre PC1 (PCA)", ylabel="Índice de Oja",
                title=f"Oja vs. PCA  (r = {np.corrcoef(index, pc1_scores)[0, 1]:.4f})")
    _scatter_by_topic(axes[1], followers, index, topic)
    axes[1].set_xscale("log")
    r = np.corrcoef(index, np.log10(followers))[0, 1]
    axes[1].set(xlabel="followers (escala log)", ylabel="Índice de Oja",
                title=f"Índice de Oja vs. seguidores  (r con log10 = {r:.2f})")
    _topic_legend(axes[1], loc="lower right")
    fig.tight_layout()
    return fig


def plot_strategy_comparison(table: pd.DataFrame, metric: str) -> plt.Figure:
    """Barras horizontales agrupadas: una métrica por cliente y estrategia."""
    clients = table.index.get_level_values(0).unique()
    strategies = table.index.get_level_values(1).unique()
    height = 0.8 / len(strategies)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, (color, strategy) in enumerate(zip(LINE_SERIES, strategies)):
        values = [table.loc[(client, strategy), metric] for client in clients]
        positions = np.arange(len(clients)) - 0.4 + height * (k + 0.5)
        ax.barh(positions, values, height=height * 0.9, color=color, label=strategy)
        for y, value in zip(positions, values):
            ax.text(value + 0.15, y, f"{value:.2f}", va="center", fontsize=7.5, color=INK_SECONDARY)
    ax.set_yticks(range(len(clients)), clients)
    ax.invert_yaxis()
    ax.set(xlabel=f"{metric} (menor es mejor)", title=f"{metric} según la estrategia")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_patterns(patterns, titles: list[str], ncols: int = 5, suptitle: str | None = None,
                  size: float = 1.7) -> plt.Figure:
    """Patrones ±1 como imágenes cuadradas, con +1 en negro."""
    n = len(patterns)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    lines = max(str(title).count("\n") + 1 for title in titles)
    height = (size + 0.17 * lines) * nrows + (0.4 if suptitle else 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, height), squeeze=False)
    for ax, pattern, title in zip(axes.flat, patterns, titles):
        side = int(round(np.sqrt(np.size(pattern))))
        ax.imshow(np.reshape(pattern, (side, side)), cmap="gray_r", vmin=-1, vmax=1)
        if side <= 10:
            # Separación entre píxeles para contar los que cambian en patrones chicos.
            for edge in np.arange(side - 1) + 0.5:
                ax.axhline(edge, color=GRID, linewidth=0.6)
                ax.axvline(edge, color=GRID, linewidth=0.6)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BASELINE)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    if suptitle:
        fig.suptitle(suptitle, color=INK)
    fig.tight_layout()
    return fig


def plot_rates(tables: dict[str, pd.DataFrame], xlabel: str, ylabel: str = "Proporción",
               suptitle: str | None = None, vline: tuple[float, str] | None = None,
               x_percent: bool = True, ncols: int = 4) -> plt.Figure:
    """Un panel por tabla: el índice es el eje x y cada columna, una línea (mismo color en
    todos los paneles)."""
    ncols = min(ncols, len(tables))
    nrows = int(np.ceil(len(tables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.5 * nrows), squeeze=False,
                             sharey=True)
    percent = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    for ax, (title, table) in zip(axes.flat, tables.items()):
        for color, column in zip(LINE_SERIES, table.columns):
            ax.plot(table.index, table[column], color=color, linewidth=2, marker="o",
                    markersize=4, label=column)
        if vline is not None:
            ax.axvline(vline[0], color=INK_SECONDARY, linestyle="--", linewidth=1)
            ax.text(vline[0], 0.5, f" {vline[1]}", fontsize=8, color=INK_SECONDARY)
        ax.set(title=title, xlabel=xlabel, ylim=(-0.03, 1.03))
        ax.yaxis.set_major_formatter(percent)
        if x_percent:
            ax.xaxis.set_major_formatter(percent)
    for ax in axes.flat[len(tables):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes.flat[0].legend(loc="best")
    if suptitle:
        fig.suptitle(suptitle, color=INK)
    fig.tight_layout()
    return fig


def plot_energy(curves: dict[str, np.ndarray], references: dict[str, float], title: str) -> plt.Figure:
    """Energía después de cada neurona que cambia, una línea por consulta, con niveles de
    referencia (por ejemplo, la energía de los patrones almacenados)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for color, (label, energy) in zip(LINE_SERIES, curves.items()):
        ax.plot(np.arange(len(energy)), energy, color=color, linewidth=2, label=label)
    for label, value in references.items():
        ax.axhline(value, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], value, f"{label}  ", ha="right", va="bottom", fontsize=8,
                color=INK_SECONDARY)
    ax.set(xlabel="Neuronas que cambiaron de estado", ylabel="Energía", title=title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
