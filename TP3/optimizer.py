"""Optimizadores para actualizar los pesos y bias de la red."""


def _validate_inputs(nn, gradients, learning_rate: float) -> None:
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo")
    if len(nn) != len(gradients):
        raise ValueError("nn y gradients deben tener la misma cantidad de capas")

    for layer, layer_gradients in zip(nn, gradients):
        if len(layer) != len(layer_gradients):
            raise ValueError("cada capa y sus gradientes deben tener la misma cantidad de neuronas")
        for neuron, (weight_gradient, _) in zip(layer, layer_gradients):
            if neuron.weights.shape != weight_gradient.shape:
                raise ValueError("el gradiente de pesos debe tener la misma forma que los pesos")


def _zero_state(nn):
    """Crea un estado (pesos, bias) con ceros para cada neurona."""
    return [
        [(neuron.weights * 0.0, 0.0) for neuron in layer]
        for layer in nn
    ]


def _validate_state_shape(nn, state) -> None:
    if len(nn) != len(state) or any(
        len(layer) != len(layer_state)
        for layer, layer_state in zip(nn, state)
    ):
        raise ValueError("la arquitectura de nn cambió después de inicializar el optimizador")


class SGD:
    """Descenso por gradiente estocástico clásico."""

    def __init__(self, learning_rate: float):
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        self.learning_rate = learning_rate

    def step(self, nn, gradients) -> None:
        _validate_inputs(nn, gradients, self.learning_rate)

        for layer, layer_gradients in zip(nn, gradients):
            for neuron, (weight_gradient, bias_gradient) in zip(layer, layer_gradients):
                neuron.weights -= self.learning_rate * weight_gradient
                neuron.bias -= self.learning_rate * bias_gradient


class Momentum:
    """Descenso por gradiente con momento clásico (heavy ball)."""

    def __init__(self, learning_rate: float, momentum: float = 0.9):
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        if not 0 <= momentum < 1:
            raise ValueError("momentum debe estar entre 0 y 1")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def step(self, nn, gradients) -> None:
        _validate_inputs(nn, gradients, self.learning_rate)
        if self.velocity is None:
            self.velocity = _zero_state(nn)
        _validate_state_shape(nn, self.velocity)

        for layer_state, layer, layer_gradients in zip(self.velocity, nn, gradients):
            for index, (neuron, (weight_gradient, bias_gradient)) in enumerate(
                zip(layer, layer_gradients)
            ):
                weight_velocity, bias_velocity = layer_state[index]
                weight_velocity *= self.momentum
                weight_velocity += weight_gradient
                bias_velocity = self.momentum * bias_velocity + bias_gradient
                neuron.weights -= self.learning_rate * weight_velocity
                neuron.bias -= self.learning_rate * bias_velocity
                # El vector de velocidad se actualiza in-place; el escalar debe reasignarse.
                layer_state[index] = (weight_velocity, bias_velocity)


class RMSProp:
    """RMSProp con promedio móvil de los gradientes al cuadrado."""

    def __init__(
        self,
        learning_rate: float,
        decay: float = 0.9,
        epsilon: float = 1e-8,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        if not 0 <= decay < 1:
            raise ValueError("decay debe estar entre 0 y 1")
        if epsilon <= 0:
            raise ValueError("epsilon debe ser positivo")
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.mean_square = None

    def step(self, nn, gradients) -> None:
        _validate_inputs(nn, gradients, self.learning_rate)
        if self.mean_square is None:
            self.mean_square = _zero_state(nn)
        _validate_state_shape(nn, self.mean_square)

        for layer_state, layer, layer_gradients in zip(self.mean_square, nn, gradients):
            for index, (neuron, (weight_gradient, bias_gradient)) in enumerate(
                zip(layer, layer_gradients)
            ):
                weight_avg, bias_avg = layer_state[index]
                weight_avg *= self.decay
                weight_avg += (1 - self.decay) * weight_gradient**2
                bias_avg = self.decay * bias_avg + (1 - self.decay) * bias_gradient**2
                neuron.weights -= self.learning_rate * weight_gradient / (
                    weight_avg**0.5 + self.epsilon
                )
                neuron.bias -= self.learning_rate * bias_gradient / (
                    bias_avg**0.5 + self.epsilon
                )
                layer_state[index] = (weight_avg, bias_avg)


class Adam:
    """Adam con corrección de sesgo de los promedios móviles."""

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate debe ser positivo")
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError("beta1 y beta2 deben estar entre 0 y 1")
        if epsilon <= 0:
            raise ValueError("epsilon debe ser positivo")
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment = None
        self.second_moment = None
        self.step_count = 0

    def step(self, nn, gradients) -> None:
        _validate_inputs(nn, gradients, self.learning_rate)
        if self.first_moment is None:
            self.first_moment = _zero_state(nn)
            self.second_moment = _zero_state(nn)
        _validate_state_shape(nn, self.first_moment)
        _validate_state_shape(nn, self.second_moment)
        self.step_count += 1

        correction1 = 1 - self.beta1**self.step_count
        correction2 = 1 - self.beta2**self.step_count
        for layer, layer_gradients, first_layer, second_layer in zip(
            nn, gradients, self.first_moment, self.second_moment
        ):
            for index, (neuron, (weight_gradient, bias_gradient)) in enumerate(
                zip(layer, layer_gradients)
            ):
                first = first_layer[index]
                second = second_layer[index]
                weight_mean, bias_mean = first
                weight_square, bias_square = second

                weight_mean *= self.beta1
                weight_mean += (1 - self.beta1) * weight_gradient
                bias_mean = self.beta1 * bias_mean + (1 - self.beta1) * bias_gradient
                weight_square *= self.beta2
                weight_square += (1 - self.beta2) * weight_gradient**2
                bias_square = self.beta2 * bias_square + (1 - self.beta2) * bias_gradient**2

                corrected_weight_mean = weight_mean / correction1
                corrected_bias_mean = bias_mean / correction1
                corrected_weight_square = weight_square / correction2
                corrected_bias_square = bias_square / correction2
                neuron.weights -= self.learning_rate * corrected_weight_mean / (
                    corrected_weight_square**0.5 + self.epsilon
                )
                neuron.bias -= self.learning_rate * corrected_bias_mean / (
                    corrected_bias_square**0.5 + self.epsilon
                )

                first_layer[index] = (weight_mean, bias_mean)
                second_layer[index] = (weight_square, bias_square)
