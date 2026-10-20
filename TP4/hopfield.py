"""Red de Hopfield discreta (estados ±1) con actualización asincrónica."""

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np
import pandas as pd

RULES = ("hebb", "pseudoinversa")


@dataclass(slots=True)
class Recall:
    """Resultado de una consulta a la red."""

    state: np.ndarray
    energy: np.ndarray  # energía al inicio y después de cada neurona que cambió de estado
    sweeps: int  # pasadas completas sobre las neuronas hasta que ninguna cambia
    converged: bool


@dataclass(slots=True)
class Hopfield:
    """Memoria asociativa con pesos W fijos y energía E(s) = −½·sᵀWs.

    - Regla de Hebb: W = (1/N)·Σ ξξᵀ. Sólo funciona si los patrones son casi ortogonales.
    - Regla de la pseudoinversa: W = Pᵀ(PPᵀ)⁻¹P, la proyección sobre el espacio de los
      patrones. Cada patrón almacenado queda como punto fijo aunque estén correlacionados,
      mientras sean linealmente independientes.

    En ambas la diagonal se anula: sin autoconexiones, la actualización asincrónica nunca
    aumenta la energía y la red siempre converge a un punto fijo.
    """

    rule: str = "hebb"
    weights_: np.ndarray = field(init=False, repr=False)
    patterns_: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"rule debe ser una de {RULES}")

    def fit(self, patterns: np.ndarray) -> "Hopfield":
        """`patterns`: (M, N) con valores ±1, un patrón por fila."""
        P = np.asarray(patterns, dtype=float).reshape(len(patterns), -1)
        if self.rule == "hebb":
            W = P.T @ P / P.shape[1]
        else:
            W = P.T @ np.linalg.solve(P @ P.T, P)
        np.fill_diagonal(W, 0.0)
        self.weights_ = W
        self.patterns_ = P
        return self

    def energy(self, state: np.ndarray) -> float:
        s = np.ravel(state).astype(float)
        return float(-0.5 * s @ self.weights_ @ s)

    def recall(self, state: np.ndarray, max_sweeps: int = 100, seed: int = 0) -> Recall:
        """Actualiza una neurona por vez, en orden aleatorio, hasta llegar a un punto fijo.

        Con campo local nulo la neurona conserva su estado. Cada cambio baja la energía en
        |2·hᵢ|, así que se registra sin recalcular E completa.
        """
        rng = np.random.default_rng(seed)
        s = np.ravel(state).astype(float).copy()
        energy = [self.energy(s)]
        for sweep in range(1, max_sweeps + 1):
            changed = False
            for i in rng.permutation(len(s)):
                h = self.weights_[i] @ s
                if h * s[i] < 0:
                    s[i] = -s[i]
                    energy.append(energy[-1] - 2 * abs(h))
                    changed = True
            if not changed:
                return Recall(s.reshape(np.shape(state)), np.array(energy), sweep - 1, True)
        return Recall(s.reshape(np.shape(state)), np.array(energy), max_sweeps, False)

    def stable_fraction(self) -> np.ndarray:
        """Fracción de neuronas de cada patrón almacenado que ya tienen el signo correcto.

        Un patrón es punto fijo sólo si vale 1: con Hebb el ruido cruzado de los demás
        patrones puede invertir el campo local de algunas neuronas.
        """
        fields = self.patterns_ @ self.weights_
        return np.mean(fields * self.patterns_ > 0, axis=1)


def fixed_points(model: Hopfield, chunk: int = 2**20) -> np.ndarray:
    """Todos los puntos fijos, recorriendo los 2ᴺ estados. Sólo es viable con N ≲ 25."""
    n = model.weights_.shape[0]
    W = model.weights_.astype(np.float32)
    bits = np.arange(n)
    found = []
    for start in range(0, 2**n, chunk):
        codes = np.arange(start, min(start + chunk, 2**n), dtype=np.int64)
        states = ((codes[:, None] >> bits) & 1).astype(np.float32) * 2 - 1
        found.append(states[np.all(states @ W * states >= 0, axis=1)])
    return np.concatenate(found).astype(float)


def mixture(state: np.ndarray, patterns: np.ndarray, names: list[str], size: int = 3) -> str | None:
    """Si el estado es sign(±ξᵃ ± ξᵇ ± ξᶜ) para algún trío de patrones, lo describe."""
    s = np.ravel(state)
    P = np.asarray(patterns, dtype=float).reshape(len(patterns), -1)
    for group in combinations(range(len(P)), size):
        for signs in product((1, -1), repeat=size):
            if np.array_equal(np.sign(np.array(signs) @ P[list(group)]), s):
                return " ".join(f"{'+' if sign > 0 else '−'}{names[i]}" for sign, i in zip(signs, group))
    return None


def recovery(model: Hopfield, images: dict[str, np.ndarray], damage, levels, trials: int = 20,
             decoder=None, seed: int = 0) -> pd.DataFrame:
    """Daña cada patrón almacenado con `damage(imagen, nivel, rng)` y consulta la red.

    Con `decoder` (por ejemplo, un lector de QR) también registra si la imagen dañada y la
    recuperada se leen igual que la original.
    """
    names = list(images)
    P = np.array([image.ravel() for image in images.values()])
    rng = np.random.default_rng(seed)
    rows = []
    for level in levels:
        for name, image in images.items():
            target = decoder(image) if decoder else None
            for trial in range(trials):
                damaged = damage(image, level, rng)
                result = model.recall(damaged, seed=trial)
                row = {
                    "patrón": name,
                    "nivel": level,
                    "píxeles cambiados": float(np.mean(damaged != image)),
                    "resultado": identify(result.state, P, names),
                    "pasadas": result.sweeps,
                    "cambios": len(result.energy) - 1,
                }
                if decoder:
                    row["lectura sin red"] = decoder(damaged) == target
                    row["lectura con red"] = decoder(result.state) == target
                rows.append(row)
    table = pd.DataFrame(rows)
    table["correcto"] = table["resultado"] == table["patrón"]
    return table


def overlaps(patterns: np.ndarray) -> np.ndarray:
    """Solapamiento ξᵘ·ξᵛ / N entre cada par de patrones (1 = iguales, 0 = ortogonales)."""
    P = np.asarray(patterns, dtype=float).reshape(len(patterns), -1)
    return P @ P.T / P.shape[1]


def identify(state: np.ndarray, patterns: np.ndarray, names: list[str]) -> str:
    """Nombre del patrón almacenado (o de su inverso) igual al estado; si no, 'espurio'."""
    s = np.ravel(state)
    for name, pattern in zip(names, np.asarray(patterns).reshape(len(patterns), -1)):
        if np.array_equal(s, pattern):
            return name
        if np.array_equal(s, -pattern):
            return f"inverso de {name}"
    return "espurio"
