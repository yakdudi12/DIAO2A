import numpy as np


def mse_loss(target: float, infered_value: float):
    return float(np.mean((np.asarray(target) - np.asarray(infered_value)) ** 2))


def mse_gradient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Derivada del MSE respecto de cada valor predicho."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if target.shape != prediction.shape:
        raise ValueError("target y prediction deben tener la misma forma")
    return 2.0 * (prediction - target) / prediction.size
