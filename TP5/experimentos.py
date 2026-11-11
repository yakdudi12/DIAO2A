"""Experimentos del TP5 (llamados desde los notebooks).

Todo está parametrizado por ``config.json`` (semillas, épocas, grillas) para que cada número
de los notebooks pueda regenerarse.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from datos import (SIMBOLOS, cargar_font3, cargar_pokemon, espejar, generar_simbolos,
                   ruido_sal_pimienta)
from deep import Autoencoder, VAE, bce_from_logits, pixel_errors

AQUI = Path(__file__).resolve().parent


def cargar_config():
    for ruta in (AQUI / "config.json", AQUI.parent.parent / "config.json"):
        if ruta.exists():
            return json.loads(ruta.read_text(encoding="utf-8"))
    raise FileNotFoundError("config.json")


# --------------------------------------------------------------------------- Ej. 1a
def entrenar_ae(X, enc_hidden, latent, act, opt, lr, epochs, seed, noise=None, log_every=50):
    ae = Autoencoder(X.shape[1], enc_hidden, latent, act, seed=seed)
    h = ae.fit(X, epochs=epochs, optimizer=opt, lr=lr, noise=noise, log_every=log_every)
    return ae, h


def epocas_hasta(hist, umbral=1):
    """Primera época registrada en la que el peor patrón tiene <= umbral píxeles mal."""
    for ep, _, mx in hist:
        if mx <= umbral:
            return ep
    return None


def barrido_arquitecturas(X, cfg):
    """Grilla arquitectura x activación x optimizador, con varias semillas."""
    c = cfg["ej1a"]
    filas = []
    for enc in c["arquitecturas"]:
        for act in c["activaciones"]:
            for opt, lr in c["optimizadores"].items():
                for s in c["semillas"]:
                    ae, h = entrenar_ae(X, enc, 2, act, opt, lr, c["epocas"], s)
                    err = pixel_errors(X, ae.reconstruct(X))
                    filas.append(dict(arq="-".join(map(str, enc)), act=act, opt=opt, lr=lr, seed=s,
                                      max_err=int(err.max()), mean_err=float(err.mean()),
                                      perfectos=int((err == 0).sum()), loss=h[-1][1],
                                      epocas_hasta_1=epocas_hasta(h, 1)))
    return filas


# --------------------------------------------------------------------------- Ej. 1b
def evaluar_denoising(ae, X, niveles, n_rep, seed):
    """Para cada nivel p de ruido, error de píxeles tras reconstruir (vs patrón limpio)."""
    rng = np.random.default_rng(seed)
    out = {}
    for p in niveles:
        f = ruido_sal_pimienta(p)
        err_in, err_out = [], []
        for _ in range(n_rep):
            Xr = f(X, rng)
            err_in.append((Xr != X).sum(1).mean())
            err_out.append(pixel_errors(X, ae.reconstruct(Xr)).mean())
        out[p] = (float(np.mean(err_in)), float(np.mean(err_out)))
    return out


def barrido_denoising(X, cfg):
    c = cfg["ej1b"]
    filas = []
    for latent in c["latentes"]:
        for p_tr in c["ruido_entrenamiento"]:
            for s in c["semillas"]:
                noise = ruido_sal_pimienta(p_tr) if p_tr > 0 else None
                ae, _ = entrenar_ae(X, c["arquitectura"], latent, c["activacion"], "adam", c["lr"],
                                    c["epocas"], s, noise=noise, log_every=c["epocas"])
                res = evaluar_denoising(ae, X, c["ruido_test"], c["repeticiones_test"], 10_000 + s)
                for p_te, (ein, eout) in res.items():
                    filas.append(dict(latent=latent, p_train=p_tr, p_test=p_te, seed=s,
                                      err_in=ein, err_out=eout))
    return filas


# --------------------------------------------------------------------------- Ej. 2
def datos_simbolos(cfg):
    c = cfg["ej2"]
    X, lab, sty, dsp = generar_simbolos()
    test = np.isin(sty, c["estilos_test"])
    return dict(X=X, lab=lab, sty=sty, dsp=dsp, tr=~test, te=test)


def entrenar_vae(D, cfg, beta, seed, latent=None, epocas=None):
    c = cfg["ej2"]
    v = VAE(D["X"].shape[1], c["hidden"], latent or c["latente"], c["activacion"], beta=beta, seed=seed)
    h = v.fit(D["X"][D["tr"]], epochs=epocas or c["epocas"], lr=c["lr"], batch=c["batch"],
              log_every=c["log_every"], Xval=D["X"][D["te"]])
    return v, h


def dist_vecino(muestras, ref, bloque=512):
    """Distancia euclídea al vecino más cercano de ``ref`` (sin librerías), por muestra."""
    r2 = (ref ** 2).sum(1)
    out = np.empty(len(muestras))
    for i in range(0, len(muestras), bloque):
        m = muestras[i:i + bloque]
        d2 = (m ** 2).sum(1)[:, None] - 2 * m @ ref.T + r2[None]
        out[i:i + bloque] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def entrenar_clasificador(D, cfg, seed=0, epocas=30):
    """Clasificador de símbolos (MLP con softmax, backprop propio) usado como *juez* objetivo.

    Se entrena con los estilos de entrenamiento y se valida en los estilos no vistos. Sirve para medir
    qué fracción de las muestras generadas es 'reconocible' como alguno de los 39 símbolos.
    """
    from deep import MLP, make_optimizer
    rng = np.random.default_rng(seed)
    Xtr, ytr = D["X"][D["tr"]], D["lab"][D["tr"]]
    net = MLP([Xtr.shape[1], 128, len(SIMBOLOS)], ["relu", "linear"], rng)
    opt, Y = make_optimizer("adam", 2e-3), np.eye(len(SIMBOLOS))[ytr]
    for _ in range(epocas):
        idx = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), 64):
            b = idx[s:s + 64]
            lg, c = net.forward(Xtr[b], cache=True)
            p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            g, _ = net.backward((p - Y[b]) / len(b), c)
            opt.step(net.params, g)

    def proba(X):
        lg = net.forward(X)
        p = np.exp(lg - lg.max(1, keepdims=True)); return p / p.sum(1, keepdims=True)
    acc = float((proba(D["X"][D["te"]]).argmax(1) == D["lab"][D["te"]]).mean())
    proba.feats = lambda X: np.maximum(X @ net.W[0] + net.b[0], 0.0)      # capa oculta del juez (128-d)
    return proba, acc


def distancia_frechet(F1, F2):
    """Distancia de Fréchet entre gaussianas ajustadas a dos nubes de rasgos (la métrica del FID):
    |m1-m2|² + tr(S1) + tr(S2) - 2·tr((S1 S2)^½).  Menor = distribución generada más parecida a la real."""
    m1, m2 = F1.mean(0), F2.mean(0)
    S1, S2 = np.cov(F1, rowvar=False), np.cov(F2, rowvar=False)
    ev = np.linalg.eigvals(S1 @ S2).real
    return float(((m1 - m2) ** 2).sum() + np.trace(S1) + np.trace(S2) - 2 * np.sqrt(np.clip(ev, 0, None)).sum())


def calidad_generativa(modelo, D, n=2000, seed=0, proba=None, conf=0.9):
    """Mide qué tan 'del conjunto' son las muestras generadas desde el prior N(0, I).

    * reconocibles: fracción con confianza del clasificador >= ``conf`` (referencia: imágenes reales no vistas).
    * clases: cuántos símbolos distintos aparecen entre las reconocibles (diversidad / cobertura de modos).
    * frac_copia: fracción más cercana a un ejemplo de entrenamiento que la mitad de la distancia típica real
      (una copia no es una muestra nueva).
    """
    rng = np.random.default_rng(seed)
    Xtr, Xte = D["X"][D["tr"]], D["X"][D["te"]]
    G = modelo.decode(rng.standard_normal((n, modelo.latent)))
    d_gen = dist_vecino(G, Xtr)
    d_real = dist_vecino(Xte[rng.choice(len(Xte), min(n, len(Xte)), replace=False)], Xtr)
    out = dict(d_gen_med=float(np.median(d_gen)), d_real_med=float(np.median(d_real)),
               frac_copia=float((d_gen < 0.5 * np.median(d_real)).mean()))
    if proba is not None:
        P = proba(G); rec = P.max(1) >= conf
        out.update(reconocibles=float(rec.mean()), clases=int(len(np.unique(P.argmax(1)[rec]))),
                   reconocibles_real=float((proba(Xte).max(1) >= conf).mean()),
                   fd=distancia_frechet(proba.feats(G), proba.feats(Xte)),
                   fd_ref=distancia_frechet(proba.feats(Xtr[rng.choice(len(Xtr), len(Xte), replace=False)]), proba.feats(Xte)))
    return out


def barrido_beta(D, cfg, latent=None, proba=None):
    c = cfg["ej2"]; latent = latent or c["latente"]
    filas = []
    for beta in c["betas"]:
        for s in c["semillas"]:
            v, h = entrenar_vae(D, cfg, beta, s, latent=latent)
            Xte = D["X"][D["te"]]
            rec_te = float(pixel_errors(Xte, v.reconstruct(Xte)).mean())
            q = calidad_generativa(v, D, seed=s, proba=proba)
            mu, _ = v.encode(D["X"][D["tr"]])
            filas.append(dict(latent=latent, beta=beta, seed=s, val_elbo=h[-1]["val_elbo"], rec=h[-1]["rec"], kl=h[-1]["kl"],
                              err_px_test=rec_te, **q, std_mu=float(mu.std())))
    return filas


# --------------------------------------------------------------------------- Ej. 2 con Pokémon
def datos_pokemon(cfg):
    """Sprites a ``lado``x``lado`` RGB. Se retienen ``n_validacion`` Pokémon completos (nunca vistos, ni
    espejados) para validar; el entrenamiento se duplica con el reflejo horizontal de cada sprite."""
    c = cfg["ej2_pokemon"]
    X, nombres = cargar_pokemon(c["lado"], margen=c["margen"])
    perm = np.random.default_rng(c["semilla_particion"]).permutation(len(X))
    te, tr = np.sort(perm[:c["n_validacion"]]), np.sort(perm[c["n_validacion"]:])
    Xtr = np.concatenate([X[tr], espejar(X[tr], c["lado"])]) if c["espejar"] else X[tr]
    return dict(X=X, nombres=nombres, tr=tr, te=te, Xtr=Xtr, Xte=X[te], lado=c["lado"],
                shape=(c["lado"], c["lado"], 3))


def entrenar_vae_pokemon(P, cfg, beta, seed, latent):
    c = cfg["ej2_pokemon"]
    v = VAE(P["Xtr"].shape[1], c["hidden"], latent, c["activacion"], beta=beta, seed=seed)
    h = v.fit(P["Xtr"], epochs=c["epocas"], lr=c["lr"], batch=c["batch"], log_every=c["log_every"], Xval=P["Xte"])
    return v, h


def guardar_vae(v, hist, ruta):
    np.savez_compressed(ruta, *[p.astype(np.float32) for p in v.params],
                        hist=np.array([[r[k] for k in ("epoch", "loss", "rec", "kl", "val_elbo")] for r in hist]))


def cargar_vae(ruta, d_in, hidden, latent, act, beta):
    """Reconstruye un VAE guardado con ``guardar_vae`` (los parámetros se copian en el mismo orden)."""
    v = VAE(d_in, hidden, latent, act, beta=beta, seed=0)
    with np.load(ruta) as f:
        for i, p in enumerate(v.params):
            p[...] = f[f"arr_{i}"]
        H = f["hist"]
    return v, [dict(zip(("epoch", "loss", "rec", "kl", "val_elbo"), map(float, r))) for r in H]


def vae_pokemon(P, cfg, beta, latent, carpeta, seed=None):
    """Carga el modelo de ``carpeta`` si existe; si no, lo entrena y lo guarda."""
    c = cfg["ej2_pokemon"]; seed = c["semilla_final"] if seed is None else seed
    ruta = Path(carpeta) / f"ej2p_b{beta:g}_d{latent}_s{seed}.npz"
    if ruta.exists():
        return cargar_vae(ruta, P["Xtr"].shape[1], c["hidden"], latent, c["activacion"], beta)
    v, h = entrenar_vae_pokemon(P, cfg, beta, seed, latent)
    Path(carpeta).mkdir(exist_ok=True); guardar_vae(v, h, ruta)
    return v, h


def base_pca(X, k):
    """Componentes principales con SVD de numpy: espacio de rasgos para comparar distribuciones de imágenes."""
    m = X.mean(0)
    _, _, Vt = np.linalg.svd(X - m, full_matrices=False)
    return lambda Y: (Y - m) @ Vt[:k].T


def calidad_pokemon(modelo, P, cfg, seed=0, rasgos=None):
    """¿Las muestras del prior se parecen a Pokémon reales sin ser copias?

    * fd / fd_ref: distancia de Fréchet en el espacio PCA del entrenamiento entre *n_val* muestras generadas y el
      entrenamiento, frente a la de los *n_val* Pokémon reales de validación (mismo tamaño de muestra, mismo sesgo).
      fd se promedia sobre 20 subconjuntos. fd ≈ fd_ref significa "tan parecidas al conjunto como un Pokémon nuevo".
    * d_gen_med / d_real_med: distancia euclídea mediana al vecino de entrenamiento más cercano.
    * frac_copia: fracción de muestras más cerca de un dato que la mitad de la distancia típica de un Pokémon real nuevo.
    """
    c = cfg["ej2_pokemon"]; rng = np.random.default_rng(seed)
    rasgos = rasgos or base_pca(P["Xtr"], c["componentes_pca"])
    G = modelo.decode(rng.standard_normal((c["muestras_calidad"], modelo.latent)))
    Ftr, Fte, Fg = rasgos(P["Xtr"]), rasgos(P["Xte"]), rasgos(G)
    n = len(Fte)
    fd = [distancia_frechet(Fg[rng.choice(len(Fg), n, replace=False)], Ftr) for _ in range(20)]
    d_gen, d_real = dist_vecino(G, P["Xtr"]), dist_vecino(P["Xte"], P["Xtr"])
    return dict(fd=float(np.mean(fd)), fd_ref=distancia_frechet(Fte, Ftr),
                d_gen_med=float(np.median(d_gen)), d_real_med=float(np.median(d_real)),
                frac_copia=float((d_gen < 0.5 * np.median(d_real)).mean()))


def fila_barrido_pokemon(P, cfg, beta, latent, seed, carpeta=None):
    """Una celda del barrido β x latente x semilla (la llama correr_ej2_pokemon.py en paralelo)."""
    if carpeta is not None:
        v, h = vae_pokemon(P, cfg, beta, latent, carpeta, seed=seed)
    else:
        v, h = entrenar_vae_pokemon(P, cfg, beta, seed, latent)
    mu, _ = v.encode(P["Xte"])
    rec_te = float(bce_from_logits(v.dec.forward(mu), P["Xte"]).mean())
    mae_te = float(np.abs(v.decode(mu) - P["Xte"]).mean())
    q = calidad_pokemon(v, P, cfg, seed=seed)
    mu_tr, lv_tr = v.encode(P["Xtr"])
    kl = float((0.5 * (mu_tr ** 2 + np.exp(lv_tr) - lv_tr - 1).sum(1)).mean())
    return dict(latent=latent, beta=beta, seed=seed, val_elbo=h[-1]["val_elbo"], kl=kl,
                rec_val=rec_te, mae_val=mae_te, std_mu=float(mu_tr.std()), **q)
