"""Alfabeto de 5×5 como patrones ±1 (# = 1, . = −1) y elección de letras poco solapadas."""

from itertools import combinations

import numpy as np
import pandas as pd

import noise
from hopfield import Hopfield, overlaps

# La J es la del enunciado.
_ALPHABET = {
    "A": ".###. #...# ##### #...# #...#",
    "B": "####. #...# ####. #...# ####.",
    "C": ".#### #.... #.... #.... .####",
    "D": "####. #...# #...# #...# ####.",
    "E": "##### #.... ####. #.... #####",
    "F": "##### #.... ####. #.... #....",
    "G": ".#### #.... #.### #...# .###.",
    "H": "#...# #...# ##### #...# #...#",
    "I": "##### ..#.. ..#.. ..#.. #####",
    "J": "##### ...#. ...#. #..#. ###..",
    "K": "#...# #..#. ###.. #..#. #...#",
    "L": "#.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #...# #...#",
    "N": "#...# ##..# #.#.# #..## #...#",
    "O": ".###. #...# #...# #...# .###.",
    "P": "####. #...# ####. #.... #....",
    "Q": ".###. #...# #.#.# #..#. .##.#",
    "R": "####. #...# ####. #..#. #...#",
    "S": ".#### #.... .###. ....# ####.",
    "T": "##### ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #.#.# ##.## #...#",
    "X": "#...# .#.#. ..#.. .#.#. #...#",
    "Y": "#...# .#.#. ..#.. ..#.. ..#..",
    "Z": "##### ...#. ..#.. .#... #####",
}

LETTERS = {
    letter: np.array([[1.0 if pixel == "#" else -1.0 for pixel in row] for row in rows.split()])
    for letter, rows in _ALPHABET.items()
}


def stack(letters) -> np.ndarray:
    """(M, 25) con un patrón aplanado por letra."""
    return np.array([LETTERS[letter].ravel() for letter in letters])


def most_orthogonal(k: int, letters=None) -> list[tuple[float, float, tuple[str, ...]]]:
    """Todos los grupos de `k` letras como (máx |solapamiento|, medio, letras), del más
    ortogonal al menos.

    Se ordena por el mayor |solapamiento| entre pares (lo que decide si dos letras se
    confunden) y, a igualdad, por el |solapamiento| medio.
    """
    letters = list(LETTERS) if letters is None else list(letters)
    full = np.abs(overlaps(stack(letters)))
    scored = []
    for group in combinations(range(len(letters)), k):
        pairs = full[np.ix_(group, group)][np.triu_indices(k, 1)]
        scored.append((pairs.max(), pairs.mean(), tuple(letters[i] for i in group)))
    scored.sort()
    return scored


def capacity(sizes, n_groups: int = 200, noise_fraction: float = 0.12, seed: int = 0) -> pd.DataFrame:
    """Para grupos de M letras elegidas al azar, entrenados con Hebb: fracción de letras que
    quedan como punto fijo y fracción que se recupera desde una copia con ruido."""
    rng = np.random.default_rng(seed)
    alphabet = list(LETTERS)
    rows = []
    for size in sizes:
        for _ in range(n_groups):
            P = stack(rng.choice(alphabet, size, replace=False))
            model = Hopfield().fit(P)
            recovered = [
                np.array_equal(model.recall(noise.flip(p, noise_fraction, rng)).state, p) for p in P
            ]
            rows.append({
                "M": size,
                "punto fijo": float(np.mean(model.stable_fraction() == 1)),
                "recuperada con ruido": float(np.mean(recovered)),
            })
    return pd.DataFrame(rows)
