import numpy as np


def relu(x: float):
    return max(0, x)

def sigmoid(x: float):
    return 1 / (1 + np.exp(-x))

def identity(x: float):
    return x


def derivative(activation, z: float) -> float:
    """Derivada de la activación respecto de su entrada z."""
    if activation is identity:
        return 1.0
    if activation is relu:
        return 1.0 if z > 0 else 0.0
    if activation is sigmoid:
        value = sigmoid(z)
        return value * (1.0 - value)
    raise ValueError(f"No se conoce la derivada de {activation}")
