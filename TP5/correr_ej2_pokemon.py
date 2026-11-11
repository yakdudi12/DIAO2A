"""Barrido del ejercicio 2 con Pokémon (β x latente x semilla) en procesos paralelos.

Uso:  python correr_ej2_pokemon.py [procesos]      (por defecto 6)
Guarda results/ej2p_barrido.json y, para la semilla final, los modelos β=0 y β=final de cada latente
(results/ej2p_b{β}_d{d}_s{semilla}.npz), que el notebook carga en lugar de reentrenar.
Cada proceso usa pocos hilos de BLAS: con matrices de este tamaño, muchos hilos por proceso son más lentos.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import json, sys, time
from multiprocessing import Pool
from pathlib import Path

from experimentos import cargar_config, datos_pokemon, fila_barrido_pokemon

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "results"
cfg = cargar_config(); c = cfg["ej2_pokemon"]


def celda(args):
    beta, d, s = args
    t = time.time()
    P = datos_pokemon(cfg)
    guardar = s == c["semilla_final"] and beta in (0.0, c["beta_final"])
    fila = fila_barrido_pokemon(P, cfg, beta, d, s, carpeta=OUT if guardar else None)
    print(f"β={beta:g} d={d} semilla={s}: {time.time() - t:.0f} s", flush=True)
    return fila


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tareas = [(b, d, s) for d in c["latentes"] for b in c["betas"] for s in c["semillas"]]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        filas = pool.map(celda, tareas, chunksize=1)
    (OUT / "ej2p_barrido.json").write_text(json.dumps(filas, indent=1, default=float))
    print("listo:", len(filas), "corridas")
