"""Ejercicio 2 (VAE) para un tamaño de latente.  Uso: python correr_ej2.py 16
Guarda results/ej2_beta_d{d}.json y los modelos finales (beta=1 y beta=0) de ese tamaño."""
import json, pickle, sys, time
from pathlib import Path
from experimentos import *

d = int(sys.argv[1]); AQUI = Path(__file__).resolve().parent
OUT = AQUI / "results"; OUT.mkdir(exist_ok=True)
cfg = cargar_config(); D = datos_simbolos(cfg); c = cfg["ej2"]
t0 = time.time()
proba, acc = entrenar_clasificador(D, cfg)
(OUT / "ej2_clasificador.json").write_text(json.dumps(dict(acc_estilos_no_vistos=acc)))
filas = barrido_beta(D, cfg, latent=d, proba=proba)
(OUT / f"ej2_beta_d{d}.json").write_text(json.dumps(filas, indent=1, default=float))
for beta, nom in ((c["beta_final"], "vae"), (0.0, "ae")):
    v, h = entrenar_vae(D, cfg, beta, c["semilla_final"], latent=d)
    pickle.dump(dict(params=[p.copy() for p in v.params], hist=h), open(OUT / f"ej2_{nom}_d{d}.pkl", "wb"))
print("listo d =", d, round(time.time() - t0), "s")
