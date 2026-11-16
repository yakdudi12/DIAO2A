"""Extensión del ejercicio 2b: KL-VAE con pérdida perceptual y adversarial (torch).

Es el autoencoder de la primera etapa de Latent Diffusion (Rombach et al., 2022), que reutiliza la pérdida
de VQGAN (Esser et al., 2021) con regularización KL en lugar de cuantización:

    L = Σ (|x − x̂| + w_lpips·LPIPS(x, x̂)) + w_kl·KL(q(z|x) ‖ N(0, I)) + λ·L_G
    L_D = ½ (E[relu(1 − D(x))] + E[relu(1 + D(x̂))])                        (hinge)

* LPIPS (Zhang et al., 2018) con AlexNet preentrenada en ImageNet, del paquete ``lpips``. Como AlexNet
  reduce mucho la resolución, las imágenes se amplían a ``lado_lpips`` antes de compararlas.
* D es el PatchGAN de pix2pix (Isola et al., 2017): cada salida juzga un parche.
* λ es el peso adaptativo de Esser et al.: ‖∇L_rec‖ / ‖∇L_G‖ en la última capa del decodificador, por
  ``w_disc``. El discriminador se enciende recién en la época ``inicio_disc``.
* Con w_kl = 1e-6 el posterior casi no se regulariza y N(0, I) deja de ser un buen generador: las muestras
  nuevas salen de una mezcla de gaussianas ajustada a los códigos μ de entrenamiento (``ajustar_gmm``).

La arquitectura del VAE es la de ``deep_conv.ConvVAE`` (convoluciones 4x4 de paso 2, cabezas densas μ y
log σ², decodificador con ampliación x2 + convolución 3x3), escrita en torch con diferenciación automática.
Hacia afuera recibe y devuelve filas aplanadas (N, lado*lado*3) en numpy, igual que ``ConvVAE``.

Requiere torch y lpips, que no son dependencias del proyecto: se agregan al correr, p. ej.
``uv run --with torch --with lpips python correr_klgan.py``.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def a_tensor(X, lado):
    """Filas (N, lado*lado*3) en [0, 1] -> tensor (N, 3, lado, lado) en el dispositivo."""
    t = torch.as_tensor(np.asarray(X, dtype=np.float32).reshape(-1, lado, lado, 3)).permute(0, 3, 1, 2)
    return t.contiguous().to(DISPOSITIVO)


def a_filas(T):
    return T.permute(0, 2, 3, 1).reshape(len(T), -1).detach().cpu().numpy().astype(float)


def correr_imagenes(x, dy, dx):
    """Desplaza una tanda (N, 3, L, L) dy, dx píxeles rellenando con blanco (el aumento de deep_conv)."""
    if dy == 0 and dx == 0:
        return x
    L = x.shape[-1]
    p = F.pad(x, (abs(dx), abs(dx), abs(dy), abs(dy)), value=1.0)
    return p[:, :, abs(dy) - dy:abs(dy) - dy + L, abs(dx) - dx:abs(dx) - dx + L]


class KLVAE(nn.Module):
    """La arquitectura de deep_conv.ConvVAE (sin alfa) en torch."""

    def __init__(self, lado, canales, latente):
        super().__init__()
        self.lado, self.canales, self.latent = lado, tuple(canales), latente
        self.base, self.cb = lado // 2 ** len(canales), canales[-1]
        h = self.base * self.base * self.cb
        capas, c0 = [], 3
        for c in canales:
            capas += [nn.Conv2d(c0, c, 4, 2, 1), nn.ReLU()]
            c0 = c
        self.enc = nn.Sequential(*capas, nn.Flatten())
        self.head_mu, self.head_lv = nn.Linear(h, latente), nn.Linear(h, latente)
        self.fc = nn.Linear(latente, h)
        capas = [nn.ReLU()]
        for c in list(reversed(canales[:-1])) + [canales[0]]:
            capas += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c0, c, 3, 1, 1), nn.ReLU()]
            c0 = c
        capas.append(nn.Conv2d(c0, 3, 3, 1, 1))
        self.dec = nn.Sequential(*capas)

    def codificar(self, x):
        h = self.enc(x)
        return self.head_mu(h), self.head_lv(h).clamp(-12, 6)

    def logits(self, z):
        return self.dec(self.fc(z).view(len(z), self.cb, self.base, self.base))

    def forward(self, x):
        mu, lv = self.codificar(x)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return torch.sigmoid(self.logits(z)), mu, lv

    @torch.no_grad()
    def encode(self, X, tanda=256):
        self.eval()
        mus, lvs = [], []
        for i in range(0, len(X), tanda):
            mu, lv = self.codificar(a_tensor(X[i:i + tanda], self.lado))
            mus.append(mu.cpu().numpy()); lvs.append(lv.cpu().numpy())
        return np.concatenate(mus).astype(float), np.concatenate(lvs).astype(float)

    @torch.no_grad()
    def decode(self, Z, tanda=256):
        self.eval()
        return np.concatenate([a_filas(torch.sigmoid(self.logits(torch.as_tensor(np.asarray(Z[i:i + tanda]), dtype=torch.float32,
                                                                                  device=DISPOSITIVO))))
                               for i in range(0, len(Z), tanda)])

    def reconstruct(self, X):
        return self.decode(self.encode(X)[0])


class Discriminador(nn.Module):
    """PatchGAN de pix2pix con la inicialización de VQGAN. Recibe imágenes en [-1, 1]."""

    def __init__(self, ndf=64, n_capas=3):
        super().__init__()
        capas, c0 = [nn.Conv2d(3, ndf, 4, 2, 1), nn.LeakyReLU(0.2)], ndf
        for i in range(1, n_capas + 1):
            c = ndf * min(2 ** i, 8)
            capas += [nn.Conv2d(c0, c, 4, 2 if i < n_capas else 1, 1, bias=False), nn.BatchNorm2d(c), nn.LeakyReLU(0.2)]
            c0 = c
        capas.append(nn.Conv2d(c0, 1, 4, 1, 1))
        self.red = nn.Sequential(*capas)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.red(x)


class Perceptual(nn.Module):
    """LPIPS con AlexNet (pesos congelados) sobre imágenes en [0, 1], ampliadas a ``lado`` píxeles."""

    def __init__(self, lado=128):
        super().__init__()
        import lpips
        self.red, self.lado = lpips.LPIPS(net="alex", verbose=False).eval(), lado
        for p in self.red.parameters():
            p.requires_grad_(False)

    def forward(self, a, b):
        if a.shape[-1] != self.lado:
            a = F.interpolate(a, size=self.lado, mode="bilinear", align_corners=False)
            b = F.interpolate(b, size=self.lado, mode="bilinear", align_corners=False)
        return self.red(2 * a - 1, 2 * b - 1)


def kl_normal(mu, lv):
    return 0.5 * (mu ** 2 + lv.exp() - lv - 1).sum(1)


@torch.no_grad()
def evaluar(vae, X, perc=None, tanda=128):
    """Medias por imagen decodificando μ: BCE (nats, comparable con deep_conv), L1, LPIPS y KL."""
    vae.eval()
    tot = dict(bce=0.0, l1=0.0, lpips=0.0, kl=0.0)
    for i in range(0, len(X), tanda):
        x = a_tensor(X[i:i + tanda], vae.lado)
        mu, lv = vae.codificar(x)
        lg = vae.logits(mu)
        xh = torch.sigmoid(lg)
        tot["bce"] += float(F.binary_cross_entropy_with_logits(lg, x, reduction="sum"))
        tot["l1"] += float((x - xh).abs().sum())
        tot["kl"] += float(kl_normal(mu, lv).sum())
        if perc is not None:
            tot["lpips"] += float(perc(xh, x).sum())
    return {k: v / len(X) for k, v in tot.items()}


def entrenar(vae, Xtr, Xval, epocas, lr=2e-4, tanda=64, w_kl=1e-6, w_lpips=1.0, w_disc=0.5, inicio_disc=100,
             desplazar=0, lado_lpips=128, log_every=25, seed=0, aviso=None):
    """Entrena el KL-VAE (y el discriminador desde ``inicio_disc`` si w_disc > 0). Devuelve el historial."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vae.to(DISPOSITIVO)
    perc = Perceptual(lado_lpips).to(DISPOSITIVO) if w_lpips > 0 else None
    D = Discriminador().to(DISPOSITIVO) if w_disc > 0 else None
    opt_g = torch.optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9)) if D is not None else None
    X = a_tensor(Xtr, vae.lado)
    n, hist, t0 = len(X), [], time.time()
    ultima = vae.dec[-1].weight
    for ep in range(1, epocas + 1):
        vae.train()
        con_disc = D is not None and ep >= inicio_disc
        suma = dict(nll=0.0, kl=0.0, g=0.0, d=0.0, lam=0.0)
        idx = torch.as_tensor(rng.permutation(n), device=DISPOSITIVO)
        for s in range(0, n, tanda):
            xb = X[idx[s:s + tanda]]
            if desplazar:
                xb = correr_imagenes(xb, *(int(v) for v in rng.integers(-desplazar, desplazar + 1, 2)))
            xh, mu, lv = vae(xb)
            rec = (xb - xh).abs()
            if perc is not None:
                rec = rec + w_lpips * perc(xh, xb)
            nll = rec.sum() / len(xb)
            kl = kl_normal(mu, lv).mean()
            loss = nll + w_kl * kl
            if con_disc:
                g = -D(2 * xh - 1).mean()
                gn = torch.autograd.grad(nll, ultima, retain_graph=True)[0].norm()
                gg = torch.autograd.grad(g, ultima, retain_graph=True)[0].norm()
                lam = (gn / (gg + 1e-4)).clamp(0, 1e4).detach() * w_disc
                loss = loss + lam * g
                suma["g"] += g.item(); suma["lam"] += lam.item()
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            opt_g.step()
            if con_disc:
                ld = 0.5 * (F.relu(1 - D(2 * xb - 1)).mean() + F.relu(1 + D(2 * xh.detach() - 1)).mean())
                opt_d.zero_grad(set_to_none=True)
                ld.backward()
                opt_d.step()
                suma["d"] += ld.item()
            suma["nll"] += nll.item(); suma["kl"] += kl.item()
        if ep % log_every == 0 or ep == 1 or ep == epocas:
            nb = -(-n // tanda)
            v = evaluar(vae, Xval, perc)
            hist.append(dict(epoch=ep, seg=time.time() - t0, **{k: s_ / nb for k, s_ in suma.items()},
                             **{f"{k}_val": x for k, x in v.items()}))
            if aviso:
                aviso(hist[-1])
    return hist


def ajustar_gmm(mu, k, seed=0):
    """Mezcla de k gaussianas de covarianza completa sobre los códigos μ (el prior aprendido para generar)."""
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=seed, max_iter=500).fit(mu)


def muestrear_gmm(gmm, n, seed=0):
    gmm.random_state = seed
    return gmm.sample(n)[0]


def guardar(ruta, vae, hist, extra=None):
    torch.save(dict(estado=vae.state_dict(), arq=dict(lado=vae.lado, canales=vae.canales, latente=vae.latent),
                    hist=hist, extra=extra or {}), ruta)


def cargar(ruta):
    d = torch.load(ruta, map_location=DISPOSITIVO, weights_only=False)
    vae = KLVAE(d["arq"]["lado"], d["arq"]["canales"], d["arq"]["latente"]).to(DISPOSITIVO)
    vae.load_state_dict(d["estado"])
    return vae.eval(), d["hist"], d["extra"]
