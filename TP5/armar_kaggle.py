"""Arma kaggle/tp5_ej2b_vae_conv.ipynb: un notebook autocontenido para correr correr_ej2b.py en la GPU de Kaggle.

El notebook clona el repositorio público (datos y código del ejercicio 2), escribe deep_conv.py y
correr_ej2b.py tal como están en esta carpeta, y entrena. Requiere: Accelerator = GPU, Internet = On.
Uso:  python armar_kaggle.py
"""
import json
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = "https://github.com/yakdudi12/DIAO2A.git"


def celda(tipo, texto):
    c = {"cell_type": tipo, "metadata": {}, "source": texto.splitlines(keepends=True)}
    if tipo == "code":
        c.update(execution_count=None, outputs=[])
    return c


CORRER = '''# Un bloque por GPU (T4 x2), en paralelo; se va mostrando el registro de cada uno.
import subprocess, time, json
from pathlib import Path
bloques = ["bloque_a", "bloque_b"]
procs = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} python -u correr_ej2b.py {b} > log_{b}.txt 2>&1", shell=True)
         for g, b in enumerate(bloques)]
vistos = {b: 0 for b in bloques}
while True:
    vivos = any(p.poll() is None for p in procs)
    for b in bloques:
        lineas = Path(f"log_{b}.txt").read_text(errors="replace").splitlines() if Path(f"log_{b}.txt").exists() else []
        for l in lineas[vistos[b]:]:
            print(f"[{b}] {l}", flush=True)
        vistos[b] = len(lineas)
    if not vivos:
        break
    time.sleep(20)
print("códigos de salida:", [p.returncode for p in procs])
filas = sum((json.loads(Path(f"results/ej2b_barrido_{b}.json").read_text()) for b in bloques), [])
Path("results/ej2b_barrido.json").write_text(json.dumps(filas, indent=1))
print(len(filas), "filas")'''

celdas = [
    celda("markdown", "# TP5 · Ejercicio 2b: VAE convolucional escrito a mano (GPU con cupy)\n"
                      "Entrena `correr_ej2b.py`. Toda la retropropagación está escrita a mano en `deep_conv.py`; "
                      "`cupy` sólo hace las cuentas de arreglos en la GPU, con la misma interfaz que `numpy`."),
    celda("code", f"!git clone -q --depth 1 {REPO}\n!nvidia-smi --query-gpu=name,memory.total --format=csv\n"
                  "import cupy; print('cupy', cupy.__version__)"),
    celda("code", "%%writefile deep_conv.py\n" + (AQUI / "deep_conv.py").read_text(encoding="utf-8")),
    celda("code", "%%writefile correr_ej2b.py\n" + (AQUI / "correr_ej2b.py").read_text(encoding="utf-8")),
    celda("code", CORRER),
    celda("code", "from IPython.display import Image, display\nfrom pathlib import Path\n"
                  "for p in sorted(Path('results').glob('*_s0_*.png')):\n    print(p.name); display(Image(str(p)))"),
    celda("code", "!rm -rf DIAO2A && cd results && cp ../log_*.txt . && zip -q ../ej2b_results.zip * && ls -la .. && ls | wc -l"),
]
nb = {"cells": celdas, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
(AQUI / "kaggle").mkdir(exist_ok=True)
ruta = AQUI / "kaggle" / "tp5_ej2b_vae_conv.ipynb"
ruta.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("escrito", ruta)
