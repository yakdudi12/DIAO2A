"""Prior de difusión sobre los códigos del KL-VAE: la segunda etapa de Latent Diffusion (Rombach et al., 2022).

El KL-VAE de deep_klgan.py usa w_kl = 1e-6, así que sus códigos μ no siguen N(0, I) y para generar hace falta un
prior aprendido. Hasta ahora era una mezcla de gaussianas; acá es un DDPM (Ho et al., 2020) sobre los vectores μ
(estandarizados por dimensión): una red aprende a predecir el ruido ε que se le sumó a un código en el paso t, y
para generar se parte de ruido puro y se lo quita en T pasos. Como el latente es un vector y no un mapa espacial,
la red es una MLP con bloques residuales y codificación sinusoidal del paso, no una U-Net.

El decodificador no se toca: las muestras de la difusión y las de la mezcla de gaussianas pasan por el mismo
decodificador y se miden con las mismas métricas (Fréchet, casi copias, nitidez).

Uso:  uv run --with torch --with lpips python difusion_latente.py [bloque] [pasos]
      bloque: completo o arte128; pasos de entrenamiento del denoiser (por defecto 30 000)
Lee results/klgan_<bloque>.pt; guarda results/difusion_<bloque>.json y .png.
"""
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from correr_klgan import CONFIG, OUT, datos_arte, nitidez
from deep_klgan import DISPOSITIVO, ajustar_gmm, cargar, muestrear_gmm
from experimentos import base_pca, calidad_pokemon, cargar_config, datos_pokemon

T = 1000


def esquema():
    """Varianzas β lineales de 1e-4 a 0,02 (Ho et al., 2020), α = 1 − β y ᾱ = Π α."""
    beta = torch.linspace(1e-4, 0.02, T, device=DISPOSITIVO)
    alfa = 1 - beta
    return beta, alfa, torch.cumprod(alfa, 0)


def codificar_paso(t, dim):
    mitad = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(mitad, device=t.device) / mitad)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], 1)


class Denoiser(nn.Module):
    """ε_θ(z_t, t): MLP residual; el paso t entra sumado a la representación oculta."""

    def __init__(self, d, ancho=512, bloques=4):
        super().__init__()
        self.ancho = ancho
        self.entrada = nn.Linear(d, ancho)
        self.paso = nn.Sequential(nn.Linear(ancho, ancho), nn.SiLU(), nn.Linear(ancho, ancho))
        self.bloques = nn.ModuleList(nn.Sequential(nn.LayerNorm(ancho), nn.Linear(ancho, ancho), nn.SiLU(),
                                                   nn.Linear(ancho, ancho)) for _ in range(bloques))
        self.salida = nn.Sequential(nn.LayerNorm(ancho), nn.Linear(ancho, d))

    def forward(self, z, t):
        h = self.entrada(z) + self.paso(codificar_paso(t, self.ancho))
        for b in self.bloques:
            h = h + b(h)
        return self.salida(h)


def entrenar_difusion(Z, pasos=30000, tanda=256, lr=2e-4, seed=0, log_every=2000):
    """Pérdida simple de DDPM: ‖ε − ε_θ(√ᾱ_t z₀ + √(1 − ᾱ_t) ε, t)‖² con t uniforme."""
    torch.manual_seed(seed)
    red = Denoiser(Z.shape[1]).to(DISPOSITIVO)
    opt = torch.optim.AdamW(red.parameters(), lr=lr)
    _, _, abar = esquema()
    Z = torch.as_tensor(Z, dtype=torch.float32, device=DISPOSITIVO)
    hist, suma, t0 = [], 0.0, time.time()
    for paso in range(1, pasos + 1):
        z0 = Z[torch.randint(len(Z), (tanda,), device=DISPOSITIVO)]
        t = torch.randint(T, (tanda,), device=DISPOSITIVO)
        e = torch.randn_like(z0)
        ab = abar[t][:, None]
        loss = F.mse_loss(red(ab.sqrt() * z0 + (1 - ab).sqrt() * e, t), e)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        suma += loss.item()
        if paso % log_every == 0:
            hist.append((paso, suma / log_every))
            print(f"   paso {paso}: pérdida {suma / log_every:.4f} | {time.time() - t0:.0f} s", flush=True)
            suma = 0.0
    return red, hist


@torch.no_grad()
def muestrear(red, n, d, seed=0):
    """Muestreo ancestral de DDPM: z_{t-1} = (z_t − β_t/√(1 − ᾱ_t) ε_θ) / √α_t + √β_t ruido."""
    g = torch.Generator(device=DISPOSITIVO).manual_seed(seed)
    beta, alfa, abar = esquema()
    red.eval()
    z = torch.randn(n, d, device=DISPOSITIVO, generator=g)
    for t in reversed(range(T)):
        e = red(z, torch.full((n,), t, device=DISPOSITIVO))
        z = (z - beta[t] / (1 - abar[t]).sqrt() * e) / alfa[t].sqrt()
        if t > 0:
            z = z + beta[t].sqrt() * torch.randn(n, d, device=DISPOSITIVO, generator=g)
    return z.cpu().numpy()


def figura(G, P, lado, ruta, n=12):
    import matplotlib
    matplotlib.use("Agg")
    from graficos import fila_comparada
    vec = np.array([P["Xtr"][np.argmin(((P["Xtr"] - x) ** 2).sum(1))] for x in G["difusion"][:n]])
    fig = fila_comparada([G["difusion"][:n], vec, G["gmm10"][:n]],
                         ["Difusión latente", "Vecino real más\ncercano", "GMM k=10"], (lado, lado, 3), tam=1.2)
    fig.savefig(ruta, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    nombre = sys.argv[1] if len(sys.argv) > 1 else "completo"
    b = CONFIG[nombre]
    cfg = cargar_config()
    cfg["ej2_pokemon"]["lado"] = b["lado"]
    P = datos_arte(cfg, b["lado"], b["n_val_especies"]) if b.get("datos") == "arte" else datos_pokemon(cfg)
    rasgos = base_pca(P["Xtr"], cfg["ej2_pokemon"]["componentes_pca"])
    vae, _, _ = cargar(OUT / f"klgan_{nombre}.pt")
    mu, _ = vae.encode(P["Xtr"])
    m, s = mu.mean(0), mu.std(0) + 1e-6
    print(f"=== {nombre}: {len(mu)} códigos de dimensión {mu.shape[1]} | {DISPOSITIVO}", flush=True)
    pasos = int(sys.argv[2]) if len(sys.argv) > 2 else 30000
    red, hist = entrenar_difusion((mu - m) / s, pasos=pasos, log_every=max(2000, pasos // 20))
    n = cfg["ej2_pokemon"]["muestras_calidad"]
    G = {"difusion": vae.decode(muestrear(red, n, mu.shape[1]) * s + m)}
    for k in (10, 20):
        G[f"gmm{k}"] = vae.decode(muestrear_gmm(ajustar_gmm(mu, k), n))
    res = dict(bloque=nombre, hist=hist, nitidez_real=float(np.median(nitidez(P["Xte"], b["lado"]))))
    for k, Gk in G.items():
        res[k] = dict(**calidad_pokemon(vae, P, cfg, rasgos=rasgos, G=Gk), nitidez=float(np.median(nitidez(Gk, b["lado"]))))
        print(f"{k:9} Fréchet {res[k]['fd']:.1f} (piso {res[k]['fd_ref']:.1f}) | nitidez {res[k]['nitidez']:.3f} "
              f"(real {res['nitidez_real']:.3f}) | casi copias {res[k]['frac_copia']:.3f}", flush=True)
    (OUT / f"difusion_{nombre}.json").write_text(json.dumps(res, indent=1, default=float))
    figura(G, P, b["lado"], OUT / f"difusion_{nombre}.png")
    print("listo", flush=True)
