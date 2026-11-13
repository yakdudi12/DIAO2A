"""Ejercicio 2b: entrenamiento del VAE convolucional (deep_conv.py) sobre los sprites de Pokémon.

Uso:  python correr_ej2b.py [bloque]   (GPU con cupy si está disponible; si no, numpy en CPU: muy lento)
      TP5_RAPIDO=1 python correr_ej2b.py   (prueba de humo: 2 épocas por red)

Bloque A (32x32): misma resolución, partición, latente y métricas que el ejercicio 2, para comparar la red
                  convolucional con la MLP en igualdad de condiciones. Grilla β x aumento x semilla.
Bloque B (64x64): el generador final, más grande y entrenado más tiempo.

Guarda en results/: ej2b_barrido.json (una fila por red y por punto de control), los modelos de la semilla 0
(.npz) y una vista previa de muestras (.png) por modelo.
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
for base in (AQUI, AQUI.parent.parent / "Ejercicio_2" / "codigo", Path.cwd() / "DIAO2A" / "TP5"):
    if (base / "experimentos.py").exists():          # código y datos del ejercicio 2 (no se modifican)
        sys.path.insert(0, str(base))
        break
sys.path.insert(0, str(AQUI))

from deep_conv import GPU, ConvVAE, chequeo_gradiente, guardar_conv
from experimentos import calidad_pokemon, cargar_config, datos_pokemon

RAPIDO = bool(os.environ.get("TP5_RAPIDO"))
CONFIG = {
    "bloque_a": {"lado": 32, "canales": [32, 64, 128], "latente": 32, "betas": [0.0, 1.0], "desplazar": [0, 2],
                 "semillas": [0, 1, 2], "epocas": 600, "lr": 1e-3, "tanda": 64, "calentamiento": 50, "log_every": 10},
    "bloque_b": {"lado": 64, "canales": [64, 128, 256, 512], "latente": 64, "betas": [1.0], "desplazar": [4],
                 "semillas": [0], "epocas": 3000, "lr": 5e-4, "tanda": 64, "calentamiento": 100, "log_every": 25},
}
OUT = AQUI / "results"


def vista_previa(modelo, lado, ruta, n=16, seed=7):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = modelo.decode(np.random.default_rng(seed).standard_normal((n, modelo.latent)))
    fig, axs = plt.subplots(2, n // 2, figsize=(n // 2 * 1.3, 2.7))
    for ax, g in zip(axs.ravel(), G):
        ax.imshow(g.reshape(lado, lado, 3)); ax.axis("off")
    fig.savefig(ruta, dpi=110, bbox_inches="tight"); plt.close(fig)


def medir(m, P, cfg, seed):
    """Las mismas columnas que experimentos.fila_barrido_pokemon (ejercicio 2)."""
    rec_val, _ = m.evaluar(P["Xte"])
    mu_te, _ = m.encode(P["Xte"])
    mu_tr, lv_tr = m.encode(P["Xtr"])
    kl = float((0.5 * (mu_tr ** 2 + np.exp(lv_tr) - lv_tr - 1).sum(1)).mean())
    return dict(rec_val=rec_val, kl=kl, mae_val=float(np.abs(m.decode(mu_te) - P["Xte"]).mean()),
                std_mu=float(mu_tr.std()), **calidad_pokemon(m, P, cfg, seed=seed))


def correr_bloque(nombre, filas, ruta):
    b = CONFIG[nombre]
    cfg = copy.deepcopy(cargar_config()); cfg["ej2_pokemon"]["lado"] = b["lado"]
    P = datos_pokemon(cfg)
    print(f"\n=== {nombre}: {b['lado']}x{b['lado']}, entrenamiento {P['Xtr'].shape}, validación {P['Xte'].shape}", flush=True)
    for desp in b["desplazar"]:
        for beta in b["betas"]:
            for s in (b["semillas"][:1] if RAPIDO else b["semillas"]):
                etiqueta = f"ej2b_L{b['lado']}_b{beta:g}_a{desp}_s{s}"
                if any(f["etiqueta"] == etiqueta for f in filas):
                    continue
                t = time.time()
                m = ConvVAE(b["lado"], b["canales"], b["latente"], beta=beta, seed=s)
                epocas = 2 if RAPIDO else b["epocas"]
                hist, mejor = m.fit(P["Xtr"][:128] if RAPIDO else P["Xtr"], P["Xte"], epocas, lr=b["lr"], tanda=b["tanda"],
                                    calentamiento=0 if RAPIDO else b["calentamiento"], desplazar=desp,
                                    log_every=b["log_every"],
                                    aviso=(lambda r: print(f"   época {r['epoch']:.0f}: rec {r['rec']:.0f} kl {r['kl']:.1f} | "
                                                           f"val rec {r['rec_val']:.0f} kl {r['kl_val']:.1f} | {r['seg']:.0f} s", flush=True))
                                    if nombre == "bloque_b" else None)
                final = m.copiar_params()
                ep_mejor = min(hist, key=lambda r: r["val_elbo"] if beta > 0 else r["rec_val"])["epoch"]
                for punto, params, ep in (("final", final, epocas), ("mejor_val", mejor or final, ep_mejor)):
                    m.poner_params(params)
                    filas.append(dict(etiqueta=etiqueta, bloque=nombre, lado=b["lado"], latent=b["latente"], beta=beta,
                                      desplazar=desp, seed=s, punto=punto, epoca=ep, **medir(m, P, cfg, s)))
                    if s == b["semillas"][0]:
                        guardar_conv(OUT / f"{etiqueta}_{punto}.npz", m, hist, params)
                        vista_previa(m, b["lado"], OUT / f"{etiqueta}_{punto}.png")
                ruta.write_text(json.dumps(filas, indent=1, default=float))
                f = filas[-2]
                print(f"{etiqueta}: {time.time() - t:.0f} s | final: rec_val {f['rec_val']:.0f} kl {f['kl']:.1f} "
                      f"Fréchet {f['fd']:.1f} (piso {f['fd_ref']:.1f}) copias {f['frac_copia']:.3f} | mejor val en época {ep_mejor:.0f}",
                      flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    print("GPU (cupy):", GPU, "| chequeo de gradiente:", f"{chequeo_gradiente():.2e}", flush=True)
    (OUT / "config_ej2b.json").write_text(json.dumps(CONFIG, indent=1))
    bloques = sys.argv[1:] or ["bloque_a", "bloque_b"]      # un solo bloque: para repartirlos entre dos GPU
    ruta = OUT / ("ej2b_barrido.json" if len(bloques) == 2 else f"ej2b_barrido_{bloques[0]}.json")
    filas = json.loads(ruta.read_text()) if ruta.exists() and not RAPIDO else []
    t0 = time.time()
    for nombre in bloques:
        correr_bloque(nombre, filas, ruta)
    print(f"listo: {len(filas)} filas en {time.time() - t0:.0f} s")
