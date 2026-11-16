"""Entrena el KL-VAE con LPIPS y PatchGAN (deep_klgan.py) sobre los sprites de Pokémon, por partes.

Uso:  uv run --with torch --with lpips python correr_klgan.py [bloque ...]     (por defecto, todos)

Bloques (64x64, misma partición, arquitectura y métricas que el bloque B del ejercicio 2b):
  l1        L1 + 1e-6·KL                       (base)
  lpips     L1 + LPIPS + 1e-6·KL               (aporte de la pérdida perceptual)
  completo  L1 + LPIPS + PatchGAN + 1e-6·KL    (aporte del discriminador)

Cada modelo genera de dos maneras: N(0, I) y una mezcla de k gaussianas ajustada a los μ de entrenamiento.
Guarda en results/: klgan_<bloque>.pt, klgan_<bloque>.png (vista previa) y klgan_barrido.json (métricas).
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

from deep_klgan import DISPOSITIVO, KLVAE, ajustar_gmm, entrenar, evaluar, guardar, muestrear_gmm, Perceptual
from experimentos import calidad_pokemon, cargar_config, datos_pokemon

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "results"
BASE = {"lado": 64, "canales": [64, 128, 256, 512], "latente": 64, "epocas": 1000, "lr": 2e-4, "tanda": 64,
        "w_kl": 1e-6, "inicio_disc": 200, "desplazar": 4, "lado_lpips": 128, "log_every": 25, "seed": 0,
        "k_gmm": [1, 10, 20]}
CONFIG = {
    "l1": dict(BASE, w_lpips=0.0, w_disc=0.0),
    "lpips": dict(BASE, w_lpips=1.0, w_disc=0.0),
    "completo": dict(BASE, w_lpips=1.0, w_disc=0.5),
}


def nitidez(X, lado):
    """Módulo medio del gradiente sobre la figura (la métrica del ejercicio 2c)."""
    I = X.reshape(-1, lado, lado, 3)
    g = np.zeros(I.shape[:3])
    g[:, :, :-1] += np.abs(np.diff(I, axis=2)).mean(3)
    g[:, :-1, :] += np.abs(np.diff(I, axis=1)).mean(3)
    m = I.min(3) < 0.9
    return np.array([g[i][m[i]].mean() if m[i].sum() > 20 else 0.0 for i in range(len(I))])


def vista_previa(P, vae, G_gmm, G_prior, ruta, n=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = vae.lado
    filas = [("Reales (validación)", P["Xte"][:n]), ("Reconstrucción", vae.reconstruct(P["Xte"][:n])),
             ("Muestras GMM", G_gmm[:n]), ("Muestras N(0, I)", G_prior[:n])]
    fig, axs = plt.subplots(len(filas), n, figsize=(n * 1.2, len(filas) * 1.35))
    for r, (nom, F_) in enumerate(filas):
        for c in range(n):
            axs[r, c].imshow(np.clip(F_[c].reshape(L, L, 3), 0, 1)); axs[r, c].axis("off")
        axs[r, 0].set_title(nom, fontsize=9, loc="left")
    fig.savefig(ruta, dpi=120, bbox_inches="tight"); plt.close(fig)


def correr(nombre, filas, cfg):
    b = CONFIG[nombre]
    P = datos_pokemon(cfg)
    print(f"\n=== {nombre}: {P['Xtr'].shape} | {DISPOSITIVO}", flush=True)
    t = time.time()
    vae = KLVAE(b["lado"], b["canales"], b["latente"])
    hist = entrenar(vae, P["Xtr"], P["Xte"], b["epocas"], lr=b["lr"], tanda=b["tanda"], w_kl=b["w_kl"],
                    w_lpips=b["w_lpips"], w_disc=b["w_disc"], inicio_disc=b["inicio_disc"], desplazar=b["desplazar"],
                    lado_lpips=b["lado_lpips"], log_every=b["log_every"], seed=b["seed"],
                    aviso=lambda r: print(f"   época {r['epoch']}: nll {r['nll']:.0f} kl {r['kl']:.0f} λ {r['lam']:.3f} "
                                          f"D {r['d']:.3f} | val bce {r['bce_val']:.0f} l1 {r['l1_val']:.0f} "
                                          f"lpips {r['lpips_val']:.3f} | {r['seg']:.0f} s", flush=True))
    guardar(OUT / f"klgan_{nombre}.pt", vae, hist, b)
    v = evaluar(vae, P["Xte"], Perceptual(b["lado_lpips"]).to(DISPOSITIVO))
    mu, _ = vae.encode(P["Xtr"])
    G_prior = vae.decode(np.random.default_rng(b["seed"]).standard_normal((cfg["ej2_pokemon"]["muestras_calidad"], vae.latent)))
    fila = dict(bloque=nombre, **{k: b[k] for k in ("w_lpips", "w_disc", "w_kl", "epocas")}, seg=time.time() - t,
                **{f"{k}_val": x for k, x in v.items()}, std_mu=float(mu.std()),
                nitidez_real=float(np.median(nitidez(P["Xte"], vae.lado))),
                nitidez_rec=float(np.median(nitidez(vae.reconstruct(P["Xte"]), vae.lado))),
                prior=dict(**calidad_pokemon(vae, P, cfg, seed=b["seed"], G=G_prior),
                           nitidez=float(np.median(nitidez(G_prior, vae.lado)))))
    for k in b["k_gmm"]:
        G = vae.decode(muestrear_gmm(ajustar_gmm(mu, k, b["seed"]), len(G_prior), b["seed"]))
        fila[f"gmm{k}"] = dict(**calidad_pokemon(vae, P, cfg, seed=b["seed"], G=G), nitidez=float(np.median(nitidez(G, vae.lado))))
        if k == 10:
            vista_previa(P, vae, G, G_prior, OUT / f"klgan_{nombre}.png")
    filas.append(fila)
    print(f"{nombre}: {fila['seg']:.0f} s | val bce {fila['bce_val']:.0f} lpips {fila['lpips_val']:.3f} | nitidez rec "
          f"{fila['nitidez_rec']:.3f} (real {fila['nitidez_real']:.3f}) | " +
          " | ".join(f"{g}: Fréchet {fila[g]['fd']:.1f} (piso {fila[g]['fd_ref']:.1f}) nitidez {fila[g]['nitidez']:.3f} "
                     f"copias {fila[g]['frac_copia']:.3f}" for g in ["prior"] + [f"gmm{k}" for k in b["k_gmm"]]), flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    bloques = sys.argv[1:] or list(CONFIG)
    cfg = copy.deepcopy(cargar_config())
    cfg["ej2_pokemon"]["lado"] = BASE["lado"]
    ruta = OUT / ("klgan_barrido.json" if len(bloques) == len(CONFIG) else f"klgan_barrido_{'_'.join(bloques)}.json")
    filas = []
    for nombre in bloques:
        correr(nombre, filas, cfg)
        ruta.write_text(json.dumps(filas, indent=1, default=float))
    print("listo:", len(filas), "modelos")
