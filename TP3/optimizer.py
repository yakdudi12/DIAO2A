def sgd_step(nn, gradients, learning_rate: float) -> None:
    """Actualiza pesos y bias después de calcular todos los gradientes."""
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo")

    for layer, layer_gradients in zip(nn, gradients):
        for neuron, (weight_gradient, bias_gradient) in zip(layer, layer_gradients):
            neuron.weights -= learning_rate * weight_gradient
            neuron.bias -= learning_rate * bias_gradient
