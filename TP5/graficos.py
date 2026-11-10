"""Funciones de gráfico compartidas por los notebooks y por figuras_presentacion.py.

Criterio visual (Knaflic, *Storytelling with Data*): azul ITBA = lo que hay que mirar,
gris = contexto, rojo sólo para el riesgo/falla. Escalas fijas al comparar.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

AZUL, GRIS, ROJO, ORO = "#02568A", "#B8B8B8", "#CC0000", "#E0A800"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110, "savefig.bbox": "tight"})


def mostrar_imagenes(imgs, shape, titulos=None, ncols=8, tam=1.0, cmap="gray_r", ax_title_size=9, fig=None):
    n = len(imgs)
    nrows = int(np.ceil(n / ncols))
    fig = fig or plt.figure(figsize=(ncols * tam, nrows * tam * 1.15))
    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(np.asarray(imgs[i]).reshape(shape), cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if titulos is not None:
            ax.set_title(str(titulos[i]), fontsize=ax_title_size)
    return fig


def fila_comparada(filas, nombres, shape, tam=0.75, cmap="gray_r"):
    """Varias filas de imágenes (p. ej. original / con ruido / reconstrucción)."""
    n = len(filas[0])
    fig, axs = plt.subplots(len(filas), n, figsize=(n * tam, len(filas) * tam * 1.1), squeeze=False)
    for r, (fila, nom) in enumerate(zip(filas, nombres)):
        for c in range(n):
            axs[r, c].imshow(np.asarray(fila[c]).reshape(shape), cmap=cmap, vmin=0, vmax=1)
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            for s in axs[r, c].spines.values():
                s.set_visible(False)
        axs[r, 0].set_ylabel(nom, rotation=0, ha="right", va="center", fontsize=9)
    return fig


def scatter_latente(Z, etiquetas, ax=None, color=AZUL, tam=9, extra=None):
    ax = ax or plt.gca()
    ax.scatter(Z[:, 0], Z[:, 1], c=color, s=12, zorder=3)
    for (x, y), e in zip(Z, etiquetas):
        ax.annotate(e, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=tam, color="#222")
    if extra is not None:
        ax.scatter(*extra[:, :2].T, marker="*", s=180, c=ORO, edgecolor="k", zorder=5)
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$"); ax.grid(alpha=.25)
    return ax


def grilla_decodificada(decode, shape, lim=2.5, n=12, ax=None, cmap="gray_r", rango=None):
    """Mosaico n×n con decode(z) para z en una grilla regular.

    ``lim``: cuadrado [-lim, lim]²;  ``rango=((x0,x1),(y0,y1))`` lo reemplaza (p. ej. el rango de los códigos).
    """
    (x0, x1), (y0, y1) = rango if rango is not None else ((-lim, lim), (-lim, lim))
    xs, ys = np.linspace(x0, x1, n), np.linspace(y0, y1, n)
    h, w = shape
    mos = np.zeros((n * h, n * w))
    for i, zy in enumerate(ys[::-1]):
        P = decode(np.array([[zx, zy] for zx in xs]))
        for j in range(n):
            mos[i * h:(i + 1) * h, j * w:(j + 1) * w] = P[j].reshape(shape)
    ax = ax or plt.gca()
    ax.imshow(mos, cmap=cmap, vmin=0, vmax=1, extent=(x0, x1, y0, y1), aspect="auto")
    ax.set_xlabel("$z_1$"); ax.set_ylabel("$z_2$")
    return ax


def bandas(ax, x, M, color, label, lw=2, alpha=.2, ls="-"):
    """Media ± desvío estándar entre semillas (M: semillas × x)."""
    m, s = M.mean(0), M.std(0)
    ax.plot(x, m, color=color, lw=lw, label=label, ls=ls)
    ax.fill_between(x, m - s, m + s, color=color, alpha=alpha)
