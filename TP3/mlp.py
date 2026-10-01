"""Perceptron multicapa para clasificacion de digitos (ejercicios 2 y 3)."""
import ast

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def load_digits(path):
    frame = pd.read_csv(path)
    images = np.stack(frame['image'].map(lambda text: np.asarray(ast.literal_eval(text), dtype=np.float32)))
    labels = frame['label'].to_numpy(dtype=np.int64)
    assert images.shape[1] == 784 and np.isfinite(images).all()
    assert images.min() >= 0 and images.max() <= 1
    return images, labels


class DigitMLP:
    def __init__(self, hidden, lr=0.01, optimizer='sgd', activation='relu', seed=42, class_weights=None):
        self.hidden = tuple(hidden)
        self.lr, self.optimizer, self.activation = lr, optimizer, activation
        # Peso de cada clase en la pérdida de entrenamiento (None: todas pesan lo mismo)
        self.class_weights = None if class_weights is None else np.asarray(class_weights, dtype=np.float32)
        gen = np.random.default_rng(seed)
        sizes = (784, *self.hidden, 10)
        self.weights = [gen.normal(0, np.sqrt(2 / a if activation == 'relu' else 1 / a), (a, b)).astype(np.float32)
                        for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((1, b), dtype=np.float32) for b in sizes[1:]]
        self.params = [p for pair in zip(self.weights, self.biases) for p in pair]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def _activate(self, z):
        return np.maximum(z, 0) if self.activation == 'relu' else 1 / (1 + np.exp(-np.clip(z, -40, 40)))

    def forward(self, X, cache=False):
        a = X
        activations, preactivations = [a], []
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ w + b
            a = self._activate(z)
            preactivations.append(z)
            activations.append(a)
        logits = a @ self.weights[-1] + self.biases[-1]
        shifted = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=1, keepdims=True)
        return (probs, activations, preactivations) if cache else probs

    def predict(self, X, batch_size=1024):
        return np.concatenate([self.forward(X[i:i + batch_size]).argmax(axis=1)
                               for i in range(0, len(X), batch_size)])

    def train_batch(self, X, y):
        probs, activations, preactivations = self.forward(X, cache=True)
        sample_loss = -np.log(np.maximum(probs[np.arange(len(y)), y], 1e-12))
        delta = probs.copy()
        delta[np.arange(len(y)), y] -= 1
        if self.class_weights is not None:
            sample_weights = self.class_weights[y]
            sample_loss = sample_loss * sample_weights
            delta *= sample_weights[:, None]
        loss = sample_loss.mean()
        delta /= len(y)
        grads_w, grads_b = [None] * len(self.weights), [None] * len(self.weights)
        for layer in range(len(self.weights) - 1, -1, -1):
            grads_w[layer] = activations[layer].T @ delta
            grads_b[layer] = delta.sum(axis=0, keepdims=True)
            if layer:
                delta = delta @ self.weights[layer].T
                z = preactivations[layer - 1]
                if self.activation == 'relu':
                    delta *= (z > 0)
                else:
                    a = activations[layer]
                    delta *= a * (1 - a)
        grads = [g for pair in zip(grads_w, grads_b) for g in pair]
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if self.optimizer == 'sgd':
                p -= self.lr * g
            elif self.optimizer == 'momentum':
                self.m[i] = 0.9 * self.m[i] + g
                p -= self.lr * self.m[i]
            elif self.optimizer == 'rmsprop':
                self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                p -= self.lr * g / (np.sqrt(self.v[i]) + 1e-8)
            elif self.optimizer == 'adam':
                self.m[i] = 0.9 * self.m[i] + 0.1 * g
                self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
                m_hat = self.m[i] / (1 - 0.9 ** self.t)
                v_hat = self.v[i] / (1 - 0.999 ** self.t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            else:
                raise ValueError(self.optimizer)
        return float(loss)


def augment_digits(X, gen, max_shift=2.0, max_rotation=10.0, max_zoom=0.1):
    """Aplica a cada imagen 28×28 un desplazamiento (píxeles), rotación (grados) y zoom aleatorios."""
    n = len(X)
    padded = np.pad(X.reshape(n, 28, 28), ((0, 0), (1, 1), (1, 1)))
    angle = np.deg2rad(gen.uniform(-max_rotation, max_rotation, n))[:, None, None]
    scale = gen.uniform(1 - max_zoom, 1 + max_zoom, n)[:, None, None]
    dx, dy = gen.uniform(-max_shift, max_shift, (2, n, 1, 1))
    ys, xs = np.mgrid[0:28, 0:28] - 13.5
    # Para cada píxel de salida buscamos de qué punto de la imagen original proviene
    xo, yo = xs - dx, ys - dy
    x_src = np.clip((np.cos(angle) * xo + np.sin(angle) * yo) / scale + 13.5, -1, 28)
    y_src = np.clip((-np.sin(angle) * xo + np.cos(angle) * yo) / scale + 13.5, -1, 28)
    x0 = np.clip(np.floor(x_src), -1, 27).astype(int)
    y0 = np.clip(np.floor(y_src), -1, 27).astype(int)
    fx, fy = x_src - x0, y_src - y0
    i = np.arange(n)[:, None, None]
    # Interpolación bilineal; el borde de ceros cubre lo que cae fuera de la imagen
    out = (padded[i, y0 + 1, x0 + 1] * (1 - fx) * (1 - fy) + padded[i, y0 + 1, x0 + 2] * fx * (1 - fy)
           + padded[i, y0 + 2, x0 + 1] * (1 - fx) * fy + padded[i, y0 + 2, x0 + 2] * fx * fy)
    return out.reshape(n, 784).astype(np.float32)


def balanced_class_weights(y, alpha=1.0):
    """Peso por clase inversamente proporcional a su frecuencia, elevado a alpha.

    alpha=0 no pondera y alpha=1 compensa todo el desbalance. Se normaliza para que el peso
    promedio sobre las muestras sea 1, así la escala del gradiente no cambia.
    """
    counts = np.bincount(y, minlength=10)
    weights = (len(y) / (10 * np.maximum(counts, 1))) ** alpha
    return weights / (weights * counts).sum() * len(y)


def cross_entropy(model, X, y, batch_size=1024):
    """Cross-entropy media del modelo sobre (X, y), sin actualizar pesos."""
    total = 0.0
    for i in range(0, len(y), batch_size):
        labels = y[i:i + batch_size]
        probs = model.forward(X[i:i + batch_size])
        total += -np.log(np.maximum(probs[np.arange(len(labels)), labels], 1e-12)).sum()
    return float(total / len(y))


def fit(model, X, y, epochs, seed=42, batch_size=256, X_val=None, y_val=None, restore_best=False,
        augment=None):
    """Entrena por minibatches. Con restore_best deja los pesos de la mejor época en validación.

    augment(X_batch, gen) transforma cada minibatch de entrenamiento antes de usarlo.
    """
    gen = np.random.default_rng(seed)
    history = []
    best_accuracy, best_params = -1.0, None
    for epoch in range(epochs):
        order = gen.permutation(len(y))
        losses = []
        for start in range(0, len(y), batch_size):
            batch = order[start:start + batch_size]
            X_batch = X[batch] if augment is None else augment(X[batch], gen)
            losses.append(model.train_batch(X_batch, y[batch]))
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses))}
        if X_val is not None:
            row['val_accuracy'] = accuracy_score(y_val, model.predict(X_val))
            row['val_loss'] = cross_entropy(model, X_val, y_val)
            if restore_best and row['val_accuracy'] > best_accuracy:
                best_accuracy = row['val_accuracy']
                best_params = [p.copy() for p in model.params]
        history.append(row)
    if best_params is not None:
        for p, best in zip(model.params, best_params):
            p[...] = best
    return pd.DataFrame(history)
