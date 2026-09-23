import numpy as np
from perceptron import Perceptron
from typing import Callable
from activations import identity, relu, sigmoid
from backpropagation import backward_pass, forward_pass
from loss import mse_loss
from optimizer import sgd_step


def instantiate_layers(
    layers: list[int],
    middle_activation: Callable[[float], float],
    output_activation: Callable[[float], float],
    rng: np.random.Generator | None = None,
):
    if len(layers) < 2 or any(size <= 0 for size in layers):
        raise ValueError("layers debe contener entrada y salida con tamaños positivos")
    rng = rng if rng is not None else np.random.default_rng()
    nn = []

    for idx in range(1, len(layers)):
        layer = []
        n_inputs = layers[idx - 1]
        n_neurons = layers[idx]

        activation = (
            output_activation
            if idx == len(layers) - 1
            else middle_activation
        )

        for _ in range(n_neurons):
            weights = rng.uniform(-1, 1, size=n_inputs)
            bias = rng.uniform(-1, 1)
            layer.append(
                Perceptron(
                    weights=weights,
                    bias=bias,
                    activation=activation,
                )
            )

        nn.append(layer)

    return nn

def train(nn, inputs, targets, epochs: int, learning_rate: float) -> None:
    """SGD: una actualización por muestra en cada época."""
    if len(inputs) != len(targets) or len(inputs) == 0:
        raise ValueError("inputs y targets deben tener la misma longitud no vacía")
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in zip(inputs, targets):
            activations, weighted_sums = forward_pass(nn, x)
            total_loss += mse_loss(y, activations[-1])
            gradients = backward_pass(nn, y, activations, weighted_sums)
            sgd_step(nn, gradients, learning_rate)

        if epoch == 0 or (epoch + 1) % 200 == 0:
            print(f"Época {epoch + 1:4d} | MSE: {total_loss / len(inputs):.8f}")


def main():
    LAYERS = [1, 3, 1]  # entrada, capa oculta y salida
    nn = instantiate_layers(LAYERS, relu, identity, rng=np.random.default_rng(42))

    target_function = np.tanh
    inputs = np.linspace(-1, 1, 41).reshape(-1, 1)
    targets = target_function(inputs)

    train(nn, inputs, targets, epochs=1000, learning_rate=0.01)

    for x in (-1.0, 0.0, 1.0):
        prediction = forward_pass(nn, np.array([x]))[0][-1][0]
        print(f"x={x:4.1f} | predicción={prediction:.4f} | objetivo={target_function(x):.4f}")

if __name__ == "__main__":
    main()
