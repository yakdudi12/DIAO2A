"""VAE convolucional del ejercicio 2b, escrito a mano (sin autograd, sin torch/keras).

Es la extensión de ``deep.VAE`` (ejercicio 2): misma pérdida (BCE + β·KL), mismo truco de
reparametrización, mismo Adam. Lo único que cambia es que las capas densas del codificador y del
decodificador se reemplazan por **convoluciones**, con su retropropagación escrita a mano.

Todas las operaciones son de arreglos (suma, producto de matrices, cortes). El módulo de arreglos ``xp``
es ``numpy`` en CPU o ``cupy`` en GPU: cupy implementa la misma interfaz que numpy sobre CUDA, así que
el código es el mismo y sólo cambia dónde se hacen las cuentas. No hay diferenciación automática.

Convenciones: imágenes en formato (N, alto, ancho, canales); hacia afuera el modelo recibe y devuelve
filas aplanadas (N, lado*lado*3) en numpy, igual que ``deep.VAE``.
"""
from __future__ import annotations

import os
import time

import numpy as np

try:
    if os.environ.get("TP5_CPU"):
        raise ImportError
    import cupy as xp
    xp.cuda.runtime.getDeviceCount()
    xp.zeros(1) + 1
    GPU = True
except Exception:
    xp = np
    GPU = False


def a_numpy(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)


def sigmoid(z):
    return 1.0 / (1.0 + xp.exp(-xp.clip(z, -60, 60)))


def bce_logits(z, x):
    """BCE estable desde logits, suma por muestra (igual que deep.bce_from_logits)."""
    return (xp.maximum(z, 0) - z * x + xp.log1p(xp.exp(-xp.abs(z)))).sum(1)


# --------------------------------------------------------------------------- capas
class Conv:
    """Convolución 2-D. W: (k, k, c_entrada, c_salida).

    La convolución se escribe como k·k productos de matrices: para cada posición (i, j) del núcleo,
    la entrada desplazada (tomada cada ``paso`` píxeles) se multiplica por W[i, j] y se acumula.
    La retropropagación recorre las mismas k·k posiciones:
        dL/dW[i, j] = entrada_desplazadaᵀ · δ          dL/d entrada_desplazada += δ · W[i, j]ᵀ
    """

    def __init__(self, c_in, c_out, k, paso, pad, rng, dt):
        self.W = xp.asarray(rng.normal(0.0, np.sqrt(2.0 / (k * k * c_in)), (k, k, c_in, c_out)), dtype=dt)  # He
        self.b = xp.zeros(c_out, dtype=dt)
        self.k, self.paso, self.pad = k, paso, pad
        self.params, self.grads = [self.W, self.b], [None, None]

    def _cortes(self, Ho, Wo):
        s = self.paso
        for i in range(self.k):
            for j in range(self.k):
                yield i, j, slice(i, i + s * Ho, s), slice(j, j + s * Wo, s)

    def forward(self, x):
        p = self.pad
        xr = xp.pad(x, ((0, 0), (p, p), (p, p), (0, 0))) if p else x
        Ho, Wo = (xr.shape[1] - self.k) // self.paso + 1, (xr.shape[2] - self.k) // self.paso + 1
        out = xp.zeros((x.shape[0], Ho, Wo, self.W.shape[3]), dtype=x.dtype) + self.b
        for i, j, fy, fx in self._cortes(Ho, Wo):
            out += xr[:, fy, fx, :] @ self.W[i, j]
        self.cache = (xr, Ho, Wo)
        return out

    def backward(self, d):
        xr, Ho, Wo = self.cache
        c_in, c_out = self.W.shape[2:]
        gW, dx, d2 = xp.empty_like(self.W), xp.zeros_like(xr), d.reshape(-1, c_out)
        for i, j, fy, fx in self._cortes(Ho, Wo):
            gW[i, j] = xr[:, fy, fx, :].reshape(-1, c_in).T @ d2
            dx[:, fy, fx, :] += d @ self.W[i, j].T
        self.grads = [gW, d.sum((0, 1, 2))]
        p = self.pad
        return dx[:, p:-p, p:-p, :] if p else dx


class Ampliar:
    """Duplica alto y ancho repitiendo cada píxel (vecino más cercano). Su gradiente suma cada bloque 2x2."""
    params, grads = [], []

    def forward(self, x):
        return x.repeat(2, axis=1).repeat(2, axis=2)

    def backward(self, d):
        n, h, w, c = d.shape
        return d.reshape(n, h // 2, 2, w // 2, 2, c).sum((2, 4))


class ReLU:
    params, grads = [], []

    def forward(self, x):
        self.m = x > 0
        return x * self.m

    def backward(self, d):
        return d * self.m


class Densa:
    def __init__(self, n_in, n_out, rng, dt, he=False):
        s = np.sqrt((2.0 if he else 1.0) / n_in)
        self.W = xp.asarray(rng.normal(0.0, s, (n_in, n_out)), dtype=dt)
        self.b = xp.zeros(n_out, dtype=dt)
        self.params, self.grads = [self.W, self.b], [None, None]

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, d):
        self.grads = [self.x.T @ d, d.sum(0)]
        return d @ self.W.T


class Adam:
    """El mismo Adam de deep.py, sobre arreglos de ``xp``."""

    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = [xp.zeros_like(p) for p in params]
        self.v = [xp.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (xp.sqrt(v / c2) + self.eps)


# --------------------------------------------------------------------------- VAE convolucional
class ConvVAE:
    """q(z|x) = N(μ, diag σ²);  p(x|z) = Bernoulli(sigmoide(dec(z)));  pérdida = BCE + β·KL.

    Codificador: una convolución 4x4 de paso 2 + ReLU por cada entrada de ``canales`` (cada una divide el
    lado por 2), y dos cabezas densas (μ y log σ²).
    Decodificador: densa + ReLU, y por cada nivel [ampliar x2, convolución 3x3, ReLU]; al final una
    convolución 3x3 que devuelve los logits de los 3 canales.
    """

    def __init__(self, lado, canales, latente, beta=1.0, seed=0, dt="float32"):
        rng = np.random.default_rng(seed)
        self.lado, self.canales, self.latent, self.beta, self.dt = lado, tuple(canales), latente, beta, dt
        self.d_in, self.rng = lado * lado * 3, rng
        self.base, self.cb = lado // 2 ** len(canales), canales[-1]
        h = self.base * self.base * self.cb
        self.enc, c0 = [], 3
        for c in canales:
            self.enc += [Conv(c0, c, 4, 2, 1, rng, dt), ReLU()]
            c0 = c
        self.head_mu, self.head_lv = Densa(h, latente, rng, dt), Densa(h, latente, rng, dt)
        self.fc = Densa(latente, h, rng, dt, he=True)
        self.dec = [ReLU()]
        for c in list(reversed(canales[:-1])) + [canales[0]]:
            self.dec += [Ampliar(), Conv(c0, c, 3, 1, 1, rng, dt), ReLU()]
            c0 = c
        self.dec += [Conv(c0, 3, 3, 1, 1, rng, dt)]
        self._capas = self.enc + [self.head_mu, self.head_lv, self.fc] + self.dec

    @property
    def params(self):
        return [p for c in self._capas for p in c.params]

    def _grads(self):
        return [g for c in self._capas for g in c.grads]

    # -- pasadas internas (arreglos de xp)
    def _codificar(self, x):
        h = x.reshape(-1, self.lado, self.lado, 3)
        for c in self.enc:
            h = c.forward(h)
        self._forma_h = h.shape
        hf = h.reshape(len(x), -1)
        return self.head_mu.forward(hf), self.head_lv.forward(hf)

    def _decodificar(self, z):
        g = self.fc.forward(z).reshape(len(z), self.base, self.base, self.cb)
        for c in self.dec:
            g = c.forward(g)
        return g.reshape(len(z), -1)

    def loss_grads(self, x, beta=None, eps=None):
        beta = self.beta if beta is None else beta
        n = len(x)
        mu, lv_raw = self._codificar(x)
        lv = xp.clip(lv_raw, -12, 6)
        if eps is None:
            eps = xp.asarray(self.rng.standard_normal(mu.shape), dtype=self.dt)
        z = mu + xp.exp(0.5 * lv) * eps                      # reparametrización
        lg = self._decodificar(z)
        rec = bce_logits(lg, x)
        kl = 0.5 * (mu ** 2 + xp.exp(lv) - lv - 1).sum(1)
        loss = (rec + beta * kl).mean()
        # --- gradientes (misma cuenta que deep.VAE, capa por capa hacia atrás)
        d = ((sigmoid(lg) - x) / n).reshape(n, self.lado, self.lado, 3)
        for c in reversed(self.dec):
            d = c.backward(d)
        dz = self.fc.backward(d.reshape(n, -1))
        dmu = dz + beta * mu / n
        dlv = dz * eps * 0.5 * xp.exp(0.5 * lv) + beta * 0.5 * (xp.exp(lv) - 1) / n
        dlv = dlv * ((lv_raw > -12) & (lv_raw < 6))
        d = (self.head_mu.backward(dmu) + self.head_lv.backward(dlv)).reshape(self._forma_h)
        for c in reversed(self.enc):
            d = c.backward(d)
        return float(loss), float(rec.mean()), float(kl.mean()), self._grads()

    # -- interfaz hacia afuera (numpy, por tandas), igual que deep.VAE
    def encode(self, X, tanda=256):
        mus, lvs = [], []
        for i in range(0, len(X), tanda):
            mu, lv = self._codificar(xp.asarray(X[i:i + tanda], dtype=self.dt))
            mus.append(a_numpy(mu)); lvs.append(a_numpy(xp.clip(lv, -12, 6)))
        return np.concatenate(mus).astype(float), np.concatenate(lvs).astype(float)

    def decode(self, Z, tanda=256):
        return np.concatenate([a_numpy(sigmoid(self._decodificar(xp.asarray(Z[i:i + tanda], dtype=self.dt))))
                               for i in range(0, len(Z), tanda)]).astype(float)

    def reconstruct(self, X):
        return self.decode(self.encode(X)[0])

    def evaluar(self, X, tanda=256):
        """(BCE de reconstrucción decodificando μ, KL) medios, en nats por imagen."""
        rec = kl = 0.0
        for i in range(0, len(X), tanda):
            x = xp.asarray(X[i:i + tanda], dtype=self.dt)
            mu, lv = self._codificar(x)
            lv = xp.clip(lv, -12, 6)
            rec += float(bce_logits(self._decodificar(mu), x).sum())
            kl += float((0.5 * (mu ** 2 + xp.exp(lv) - lv - 1)).sum())
        return rec / len(X), kl / len(X)

    def copiar_params(self):
        return [a_numpy(p).astype(np.float32) for p in self.params]

    def poner_params(self, lista):
        for p, q in zip(self.params, lista):
            p[...] = xp.asarray(q, dtype=self.dt)

    def fit(self, X, Xval, epocas, lr=1e-3, tanda=64, calentamiento=0, desplazar=0, log_every=10, aviso=None):
        """Entrena con Adam. Devuelve (historial, parámetros de la época con mejor −ELBO de validación).

        calentamiento: épocas en las que β sube linealmente de 0 a su valor (evita el colapso del posterior).
        desplazar: aumento de datos; cada tanda se corre un número entero de píxeles al azar en [-d, d]
                   (rellenando con blanco). Un sprite corrido sigue siendo el mismo Pokémon.
        """
        Xd = xp.asarray(X, dtype=self.dt)
        opt, n, hist = Adam(self.params, lr), len(X), []
        mejor, mejor_p, t0 = np.inf, None, time.time()
        for ep in range(1, epocas + 1):
            beta = self.beta * min(1.0, ep / calentamiento) if calentamiento else self.beta
            idx = xp.asarray(self.rng.permutation(n))
            for s in range(0, n, tanda):
                xb = Xd[idx[s:s + tanda]]
                if desplazar:
                    xb = correr_imagenes(xb, self.lado, *self.rng.integers(-desplazar, desplazar + 1, 2))
                loss, rec, kl, g = self.loss_grads(xb, beta)
                opt.step(self.params, g)
            if ep % log_every == 0 or ep == 1 or ep == epocas:
                rv, kv = self.evaluar(Xval)
                hist.append(dict(epoch=ep, loss=loss, rec=rec, kl=kl, rec_val=rv, kl_val=kv, val_elbo=rv + kv,
                                 seg=time.time() - t0))
                criterio = rv + kv if self.beta > 0 else rv      # con β=0 el KL no se optimiza
                if ep >= calentamiento and criterio < mejor:
                    mejor, mejor_p = criterio, self.copiar_params()
                if aviso:
                    aviso(hist[-1])
        return hist, mejor_p


def correr_imagenes(x, lado, dy, dx):
    """Desplaza una tanda de imágenes aplanadas (dy, dx) píxeles, rellenando con blanco."""
    dy, dx = int(dy), int(dx)
    if dy == 0 and dx == 0:
        return x
    im = x.reshape(-1, lado, lado, 3)
    out = xp.ones_like(im)
    out[:, max(dy, 0):lado + min(dy, 0), max(dx, 0):lado + min(dx, 0)] = \
        im[:, max(-dy, 0):lado + min(-dy, 0), max(-dx, 0):lado + min(-dx, 0)]
    return out.reshape(len(x), -1)


def guardar_conv(ruta, modelo, hist, params=None):
    claves = sorted(hist[0])
    np.savez_compressed(ruta, *(params or modelo.copiar_params()),
                        hist=np.array([[r[k] for k in claves] for r in hist]), claves=np.array(claves),
                        arq=np.array([modelo.lado, modelo.latent, *modelo.canales]), beta=modelo.beta)


def cargar_conv(ruta):
    with np.load(ruta) as f:
        arq = [int(a) for a in f["arq"]]
        m = ConvVAE(arq[0], arq[2:], arq[1], beta=float(f["beta"]))
        m.poner_params([f[f"arr_{i}"] for i in range(len(m.params))])
        hist = [dict(zip([str(k) for k in f["claves"]], map(float, r))) for r in f["hist"]]
    return m, hist


def chequeo_gradiente(eps=1e-5, seed=0, por_param=8):
    """Diferencias finitas contra la retropropagación a mano (float64, ruido de reparametrización fijo).
    Devuelve el error relativo máximo sobre ``por_param`` entradas al azar de cada tensor de parámetros."""
    rng = np.random.default_rng(seed)
    m = ConvVAE(8, (3, 4), 3, beta=0.7, seed=seed, dt="float64")
    x = xp.asarray(rng.random((4, 8 * 8 * 3)))
    e = xp.asarray(rng.standard_normal((4, 3)))
    g = m.loss_grads(x, eps=e)[3]
    peor = 0.0
    for p, gp in zip(m.params, g):
        for _ in range(por_param):
            i = tuple(int(rng.integers(0, s)) for s in p.shape)
            v = float(p[i])
            p[i] = v + eps; lp = m.loss_grads(x, eps=e)[0]
            p[i] = v - eps; lm = m.loss_grads(x, eps=e)[0]
            p[i] = v
            num = (lp - lm) / (2 * eps)
            peor = max(peor, abs(num - float(gp[i])) / max(1e-8, abs(num) + abs(float(gp[i]))))
    return peor
