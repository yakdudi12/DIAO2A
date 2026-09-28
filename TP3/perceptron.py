from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from activations import identity


@dataclass(slots=True)
class Perceptron:
    """Una neurona con un peso por cada entrada que recibe."""

    weights: np.ndarray
    bias: float = 0.0
    activation: Callable[[float], float] = field(default=identity, repr=False)

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.ndim != 1 or self.weights.size == 0:
            raise ValueError("weights debe ser un vector no vacío")

    def weighted_sum(self, inputs: np.ndarray) -> float:
        inputs = np.asarray(inputs, dtype=float)
        if inputs.shape != self.weights.shape:
            raise ValueError(
                f"Se esperaban {self.weights.size} entradas, se recibieron {inputs.shape}"
            )
        return float(np.dot(inputs, self.weights) + self.bias)

    def forward(self, inputs: np.ndarray) -> float:
        return self.activation(self.weighted_sum(inputs))
