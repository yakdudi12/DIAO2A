"""Descarga el arte oficial de PokeAPI (frente y frente shiny, 475x475 RGBA) a data/pokemon_arte/.

Uso:  python descargar_arte.py [n_especies]      (por defecto 1025, todas las especies)
Archivos: 0001.png, 0001_shiny.png, ... Lo ya descargado no se vuelve a bajar. Fuente:
https://github.com/PokeAPI/sprites (sprites/pokemon/other/official-artwork). Uso académico.
"""
import sys
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/"
DESTINO = Path(__file__).resolve().parent / "data" / "pokemon_arte"


def bajar(tarea):
    url, ruta = tarea
    if ruta.exists():
        return "ya estaba"
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                datos = r.read()
            parcial = ruta.with_suffix(".part")
            parcial.write_bytes(datos)
            parcial.rename(ruta)
            return "descargado"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return "no existe"
        except OSError:
            pass
    return "falló"


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1025
    DESTINO.mkdir(parents=True, exist_ok=True)
    tareas = [(f"{BASE}{i}.png", DESTINO / f"{i:04d}.png") for i in range(1, n + 1)]
    tareas += [(f"{BASE}shiny/{i}.png", DESTINO / f"{i:04d}_shiny.png") for i in range(1, n + 1)]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(bajar, tareas))
    print(dict(Counter(res)), "|", len(list(DESTINO.glob("*.png"))), "imágenes en", DESTINO)
    for (url, _), r in zip(tareas, res):
        if r == "falló":
            print("falló:", url)
