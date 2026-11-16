"""¿Es continuo el espacio latente del KL-VAE? Interpolaciones entre Pokémon de validación y tamaño de σ.

Si el latente es continuo, decodificar el camino recto entre los códigos μ de dos Pokémon da una transición gradual
de uno al otro; si los códigos son puntos aislados (σ mucho menor que la distancia entre ellos), los puntos
intermedios caen donde el decodificador nunca entrenó y salen imágenes rotas.

Uso:  uv run --with torch --with lpips python interpolar.py bloque [bloque ...]      (p. ej. arte128)
Guarda results/interpolacion_<bloque>.png e imprime σ frente a la dispersión de los códigos.
"""
import sys

import numpy as np

from correr_klgan import CONFIG, OUT, datos_arte
from deep_klgan import cargar
from experimentos import cargar_config, datos_pokemon

PARES = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]


def interpolar(nombre, pasos=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = CONFIG[nombre]
    cfg = cargar_config()
    cfg["ej2_pokemon"]["lado"] = b["lado"]
    P = datos_arte(cfg, b["lado"], b["n_val_especies"]) if b.get("datos") == "arte" else datos_pokemon(cfg)
    vae, _, _ = cargar(OUT / f"klgan_{nombre}.pt")
    L = b["lado"]
    X = P["X"][[i for i in P["te"] if "_shiny" not in P["nombres"][i]]]
    mu, lv = vae.encode(X)
    mu_tr, lv_tr = vae.encode(P["Xtr"])
    sigma = np.exp(0.5 * lv_tr)
    d_vecino = np.sort(np.linalg.norm(mu_tr[:500, None] - mu_tr[None, :500], axis=2), 1)[:, 1]
    print(f"{nombre}: σ mediano {np.median(sigma):.4f} | desvío de los μ {mu_tr.std():.3f} | distancia mediana al "
          f"código vecino {np.median(d_vecino):.3f} | radio típico de la nube ‖σ‖ {np.median(np.linalg.norm(sigma, axis=1)):.3f}",
          flush=True)
    t = np.linspace(0, 1, pasos)
    fig, axs = plt.subplots(len(PARES), pasos + 2, figsize=((pasos + 2) * 1.15, len(PARES) * 1.2))
    for r, (a, c) in enumerate(PARES):
        imgs = [X[a]] + list(vae.decode((1 - t[:, None]) * mu[a] + t[:, None] * mu[c])) + [X[c]]
        for k, im in enumerate(imgs):
            axs[r, k].imshow(np.clip(im.reshape(L, L, 3), 0, 1))
            axs[r, k].axis("off")
    axs[0, 0].set_title("real A", fontsize=8)
    axs[0, -1].set_title("real B", fontsize=8)
    for k, tk in enumerate(t):
        axs[0, k + 1].set_title(f"t={tk:.2f}", fontsize=8)
    fig.suptitle(f"{nombre} (w_kl = {b['w_kl']:g}): decodificación de (1 − t)·μ_A + t·μ_B", y=1.01)
    fig.savefig(OUT / f"interpolacion_{nombre}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for nombre in sys.argv[1:] or ["arte128"]:
        interpolar(nombre)
