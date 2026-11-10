"""Núcleo de aprendizaje profundo del TP5, 100 % numpy (sin autograd, sin torch).

Contenido
---------
* ``MLP``            perceptrón multicapa con retropropagación escrita a mano.
* ``SGD/Momentum/Adam``  optimizadores propios.
* ``Autoencoder``    codificador + decodificador simétricos, pérdida BCE.
* ``VAE``            autoencoder variacional (truco de reparametrización, KL cerrada).

Convenciones: los datos son filas (N, D); los pesos W tienen forma (entrada, salida).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- activaciones
ACT = {
    "linear": (lambda z: z, lambda z, a: np.ones_like(z)),
    "tanh": (np.tanh, lambda z, a: 1.0 - a * a),
    "relu": (lambda z: np.maximum(z, 0.0), lambda z, a: (z > 0).astype(z.dtype)),
    "sigmoid": (lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))),
                lambda z, a: a * (1.0 - a)),
}


def sigmoid(z):
    return ACT["sigmoid"][0](z)


# --------------------------------------------------------------------------- optimizadores
class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Momentum:
    def __init__(self, lr=0.05, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, params, grads):
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.v):
            v *= self.beta
            v += g
            p -= self.lr * v


class Adam:
    def __init__(self, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def make_optimizer(name, lr):
    return {"sgd": SGD, "momentum": Momentum, "adam": Adam}[name](lr)


# --------------------------------------------------------------------------- MLP
class MLP:
    """Capas densas: sizes=[d0, d1, ..., dk]; acts = activación por capa (len k)."""

    def __init__(self, sizes, acts, rng):
        assert len(acts) == len(sizes) - 1
        self.sizes, self.acts = list(sizes), list(acts)
        self.W, self.b = [], []
        for i, (a, c) in enumerate(zip(sizes[:-1], sizes[1:])):
            # He para relu, Xavier para el resto (ver Goodfellow et al., 2016, §8.4)
            s = np.sqrt(2.0 / a) if acts[i] == "relu" else np.sqrt(1.0 / a)
            self.W.append(rng.normal(0.0, s, (a, c)))
            self.b.append(np.zeros(c))

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x, cache=False):
        zs, as_ = [], [x]
        a = x
        for W, b, name in zip(self.W, self.b, self.acts):
            z = a @ W + b
            a = ACT[name][0](z)
            zs.append(z)
            as_.append(a)
        return (a, (zs, as_)) if cache else a

    def backward(self, dout, cache):
        """dout = dL/d(a_última) ya con la derivada de la activación de salida NO aplicada."""
        zs, as_ = cache
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            d = d * ACT[self.acts[i]][1](zs[i], as_[i + 1])
            gW[i] = as_[i].T @ d
            gb[i] = d.sum(0)
            d = d @ self.W[i].T
        return gW + gb, d  # d = dL/dx


# --------------------------------------------------------------------------- Autoencoder
def bce_from_logits(z, x):
    """BCE estable. z = logits, x ∈ [0,1]. Devuelve suma por muestra."""
    return (np.maximum(z, 0) - z * x + np.log1p(np.exp(-np.abs(z)))).sum(1)


class Autoencoder:
    """AE simétrico. enc_hidden=[h1,h2,..] -> latente -> dec simétrico. Salida: logits."""

    def __init__(self, d_in, enc_hidden, latent, act="tanh", seed=0, dec_hidden=None):
        rng = np.random.default_rng(seed)
        dec_hidden = list(reversed(enc_hidden)) if dec_hidden is None else dec_hidden
        self.enc = MLP([d_in, *enc_hidden, latent], [act] * len(enc_hidden) + ["linear"], rng)
        self.dec = MLP([latent, *dec_hidden, d_in], [act] * len(dec_hidden) + ["linear"], rng)
        self.latent, self.d_in = latent, d_in
        self.rng = rng

    # -- inferencia
    def encode(self, x):
        return self.enc.forward(x)

    def decode_logits(self, z):
        return self.dec.forward(z)

    def decode(self, z):
        return sigmoid(self.decode_logits(z))

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    # -- entrenamiento
    def loss_grads(self, x, target):
        z, ce = self.enc.forward(x, cache=True)
        lg, cd = self.dec.forward(z, cache=True)
        n = len(x)
        loss = bce_from_logits(lg, target).mean()
        dlg = (sigmoid(lg) - target) / n  # dL/dlogits para BCE + sigmoide
        gd, dz = self.dec.backward(dlg, cd)
        ge, _ = self.enc.backward(dz, ce)
        return loss, ge + gd

    @property
    def params(self):
        return self.enc.params + self.dec.params

    def fit(self, X, epochs=3000, optimizer="adam", lr=1e-2, batch=None, noise=None,
            log_every=100, callback=None):
        """noise: callable(X, rng)->X_ruidoso para entrenar como Denoising AE (target = X limpio)."""
        opt = make_optimizer(optimizer, lr) if isinstance(optimizer, str) else optimizer
        hist = []
        n = len(X)
        for ep in range(1, epochs + 1):
            idx = np.arange(n) if batch is None else self.rng.permutation(n)
            for s in range(0, n, batch or n):
                xb = X[idx[s:s + (batch or n)]]
                xin = noise(xb, self.rng) if noise is not None else xb
                loss, g = self.loss_grads(xin, xb)
                opt.step(self.params, g)
            if ep % log_every == 0 or ep == 1:
                hist.append((ep, float(loss), pixel_errors(X, self.reconstruct(X)).max()))
                if callback:
                    callback(ep, hist[-1])
        return hist


def pixel_errors(X, P):
    """# de píxeles distintos tras umbralizar la reconstrucción en 0.5, por muestra."""
    return ((P > 0.5).astype(int) != (X > 0.5).astype(int)).sum(1)


# --------------------------------------------------------------------------- VAE
class VAE:
    """VAE gaussiano con prior N(0,I).

    q(z|x)=N(mu, diag(exp(logvar))) ; p(x|z)=Bernoulli(sigmoid(dec(z)))
    Pérdida (−ELBO) = BCE(x, x̂) + β·KL(q||p),  KL = ½ Σ(μ²+σ²−log σ²−1)
    """

    def __init__(self, d_in, hidden, latent, act="tanh", beta=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.body = MLP([d_in, *hidden], [act] * len(hidden), rng)
        h = hidden[-1]
        self.head_mu = MLP([h, latent], ["linear"], rng)
        self.head_lv = MLP([h, latent], ["linear"], rng)
        self.dec = MLP([latent, *reversed(hidden), d_in], [act] * len(hidden) + ["linear"], rng)
        self.latent, self.d_in, self.beta, self.rng = latent, d_in, beta, rng

    @property
    def params(self):
        return self.body.params + self.head_mu.params + self.head_lv.params + self.dec.params

    def encode(self, x):
        h = self.body.forward(x)
        return self.head_mu.forward(h), np.clip(self.head_lv.forward(h), -12, 6)

    def decode(self, z):
        return sigmoid(self.dec.forward(z))

    def reconstruct(self, x):
        return self.decode(self.encode(x)[0])

    def loss_grads(self, x):
        n = len(x)
        h, cb = self.body.forward(x, cache=True)
        mu, cm = self.head_mu.forward(h, cache=True)
        lv_raw, cl = self.head_lv.forward(h, cache=True)
        lv = np.clip(lv_raw, -12, 6)
        eps = self.rng.standard_normal(mu.shape)
        z = mu + np.exp(0.5 * lv) * eps  # reparametrización
        lg, cd = self.dec.forward(z, cache=True)
        rec = bce_from_logits(lg, x)
        kl = 0.5 * (mu ** 2 + np.exp(lv) - lv - 1).sum(1)
        loss = (rec + self.beta * kl).mean()
        # --- gradientes
        dlg = (sigmoid(lg) - x) / n
        gd, dz = self.dec.backward(dlg, cd)
        dmu = dz + self.beta * mu / n
        dlv = dz * eps * 0.5 * np.exp(0.5 * lv) + self.beta * 0.5 * (np.exp(lv) - 1) / n
        dlv = dlv * ((lv_raw > -12) & (lv_raw < 6))
        gm, dh1 = self.head_mu.backward(dmu, cm)
        gl, dh2 = self.head_lv.backward(dlv, cl)
        gb, _ = self.body.backward(dh1 + dh2, cb)
        return loss, float(rec.mean()), float(kl.mean()), gb + gm + gl + gd

    def fit(self, X, epochs=500, lr=2e-3, batch=64, optimizer="adam", log_every=10, Xval=None):
        opt = make_optimizer(optimizer, lr)
        n, hist = len(X), []
        for ep in range(1, epochs + 1):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch):
                loss, rec, kl, g = self.loss_grads(X[idx[s:s + batch]])
                opt.step(self.params, g)
            if ep % log_every == 0 or ep == 1:
                row = dict(epoch=ep, loss=loss, rec=rec, kl=kl)
                if Xval is not None:
                    row["val_elbo"] = self.elbo(Xval)
                hist.append(row)
        return hist

    def elbo(self, X, samples=1):
        mu, lv = self.encode(X)
        tot = 0.0
        for _ in range(samples):
            z = mu + np.exp(0.5 * lv) * self.rng.standard_normal(mu.shape)
            tot += bce_from_logits(self.dec.forward(z), X)
        rec = tot / samples
        kl = 0.5 * (mu ** 2 + np.exp(lv) - lv - 1).sum(1)
        return float((rec + kl).mean())


# --------------------------------------------------------------------------- utilidades
def gradient_check(model_fn, eps=1e-5, seed=0):
    """Chequeo numérico de gradientes de un Autoencoder pequeño (devuelve error relativo máx)."""
    rng = np.random.default_rng(seed)
    ae = Autoencoder(6, [5], 2, "tanh", seed=seed)
    x = (rng.random((4, 6)) > 0.5).astype(float)
    _, g = ae.loss_grads(x, x)
    worst = 0.0
    for p, gp in zip(ae.params, g):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            lp = ae.loss_grads(x, x)[0]
            p[i] = old - eps
            lm = ae.loss_grads(x, x)[0]
            p[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gp[i]) / max(1e-8, abs(num) + abs(gp[i])))
    return worst


def gradient_check_vae(eps=1e-5, seed=0):
    """Chequeo numérico del VAE fijando la semilla del ruido de reparametrización."""
    rng = np.random.default_rng(seed)
    v = VAE(6, [5], 2, "tanh", beta=0.7, seed=seed)
    x = (rng.random((4, 6)) > 0.5).astype(float)

    def L():
        v.rng = np.random.default_rng(123)
        return v.loss_grads(x)

    g = L()[3]
    worst = 0.0
    for p, gp in zip(v.params, g):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            lp = L()[0]
            p[i] = old - eps
            lm = L()[0]
            p[i] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - gp[i]) / max(1e-8, abs(num) + abs(gp[i])))
    return worst
