"""Deterioro de patrones binarios ±1 (imágenes con +1 = negro y −1 = blanco).

- `flip`: ruido aleatorio, invierte una fracción exacta de píxeles.
- `torn_corner`, `stain` y `scratches`: daños con forma, como los de un QR impreso. El papel
  roto o raspado queda blanco y una mancha queda negra, así que sólo cambian los píxeles
  que tenían el color contrario.
"""

import numpy as np

WHITE, BLACK = -1.0, 1.0


def flip(pattern: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    noisy = np.asarray(pattern, dtype=float).copy()
    flat = noisy.reshape(-1)
    chosen = rng.choice(flat.size, int(round(fraction * flat.size)), replace=False)
    flat[chosen] *= -1
    return noisy


def torn_corner(image: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Esquina arrancada: un triángulo blanco en una esquina al azar que cubre `fraction` del área."""
    rows, cols = image.shape
    legs = np.sqrt(2 * fraction) * np.array([rows, cols])
    r, c = np.indices(image.shape)
    flip_r, flip_c = rng.integers(0, 2, size=2)
    r = rows - 1 - r if flip_r else r
    c = cols - 1 - c if flip_c else c
    mask = (r + 0.5) / legs[0] + (c + 0.5) / legs[1] <= 1
    return np.where(mask, WHITE, image)


def stain(image: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Mancha de comida: un disco negro en una posición al azar que cubre `fraction` del área."""
    radius = np.sqrt(fraction * image.size / np.pi)
    center = rng.uniform(radius, np.array(image.shape) - radius)
    r, c = np.indices(image.shape)
    mask = (r + 0.5 - center[0]) ** 2 + (c + 0.5 - center[1]) ** 2 <= radius**2
    return np.where(mask, BLACK, image)


def scratches(image: np.ndarray, fraction: float, rng: np.random.Generator, width: float = 1.0) -> np.ndarray:
    """Rayones: segmentos blancos al azar hasta cubrir `fraction` del área."""
    rows, cols = image.shape
    r, c = np.indices(image.shape) + 0.5
    mask = np.zeros(image.shape, dtype=bool)
    while mask.mean() < fraction:
        start = rng.uniform((0, 0), (rows, cols))
        end = rng.uniform((0, 0), (rows, cols))
        direction = end - start
        # Distancia de cada píxel al segmento start-end.
        t = np.clip(((r - start[0]) * direction[0] + (c - start[1]) * direction[1])
                    / (direction @ direction), 0, 1)
        distance = np.hypot(r - start[0] - t * direction[0], c - start[1] - t * direction[1])
        mask |= distance <= width / 2
    return np.where(mask, WHITE, image)


DAMAGES = {
    "Ruido aleatorio": flip,
    "Esquina rota": torn_corner,
    "Mancha": stain,
    "Rayones": scratches,
}
