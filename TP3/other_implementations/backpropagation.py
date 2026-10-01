import numpy as np

from activations import derivative
from loss import mse_gradient
from perceptron import Perceptron


def forward_pass(nn: list[list[Perceptron]], inputs: np.ndarray):
    """Devuelve las activaciones de todas las capas y sus sumas antes de activar."""
    activations = [np.asarray(inputs, dtype=float)]
    weighted_sums = []

    for layer in nn:
        z = np.array([neuron.weighted_sum(activations[-1]) for neuron in layer])
        weighted_sums.append(z)
        activations.append(np.array([
            neuron.activation(value) for neuron, value in zip(layer, z)
        ]))

    return activations, weighted_sums


def backward_pass(nn, target, activations, weighted_sums):
    """Calcula (gradiente de pesos, gradiente de bias) para cada neurona."""
    deltas = [None] * len(nn)
    loss_gradient = mse_gradient(target, activations[-1])

    for layer_idx in range(len(nn) - 1, -1, -1):
        layer = nn[layer_idx]
        if layer_idx == len(nn) - 1:
            upstream = loss_gradient
        else:
            upstream = np.array([
                sum(next_neuron.weights[neuron_idx] * next_delta
                    for next_neuron, next_delta in zip(nn[layer_idx + 1], deltas[layer_idx + 1]))
                for neuron_idx in range(len(layer))
            ])

        deltas[layer_idx] = np.array([
            upstream[idx] * derivative(neuron.activation, weighted_sums[layer_idx][idx])
            for idx, neuron in enumerate(layer)
        ])

    return [
        [(delta * activations[layer_idx], float(delta)) for delta in deltas[layer_idx]]
        for layer_idx in range(len(nn))
    ]
