import numpy as np
from PIL import Image
from render import render


def mse(image_a, image_b):
    a = np.asarray(image_a.convert("RGB"), dtype=np.float32)
    b = np.asarray(image_b.convert("RGB"), dtype=np.float32)

    return np.mean((a - b) ** 2)

def fitness(individual, target, width, height):
    candidate = render(individual, width, height)

    error = mse(candidate, target)

    return 1.0 / (1.0 + error)
