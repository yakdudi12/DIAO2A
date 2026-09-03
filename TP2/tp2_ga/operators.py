"""Operadores del algoritmo genético requeridos por el enunciado."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SELECTION_METHODS = {
    "elite",
    "roulette",
    "universal",
    "boltzmann",
    "ranking",
    "tournament_deterministic",
    "tournament_probabilistic",
}
CROSSOVER_METHODS = {"one_point", "two_point", "uniform", "region"}
MUTATION_METHODS = {"gene", "multigene", "non_uniform", "triangle"}
SURVIVAL_METHODS = {"additive", "exclusive"}


def _fitness_probabilities(fitness: Sequence[float]) -> np.ndarray:
    errors = np.asarray(fitness, dtype=float)
    aptitude = (errors.max() - errors) / (np.ptp(errors) + 1e-12) + 0.05
    return aptitude / aptitude.sum()


def select_indices(
    fitness: Sequence[float],
    amount: int,
    method: str,
    rng: np.random.Generator,
    *,
    generation: int = 0,
    max_generations: int = 1,
    tournament_size: int = 3,
    tournament_probability: float = 0.8,
    ranking_pressure: float = 1.5,
    boltzmann_temperature: float = 1.0,
    boltzmann_min_temperature: float = 0.1,
) -> np.ndarray:
    """Implementa los siete métodos de selección solicitados."""
    if method not in SELECTION_METHODS:
        raise ValueError(f"Selección desconocida: {method}")
    errors = np.asarray(fitness, dtype=float)
    population_size = len(errors)

    if method == "elite":
        ordered = np.argsort(errors)
        return np.resize(ordered, amount)

    if method == "roulette":
        return rng.choice(population_size, amount, p=_fitness_probabilities(errors))

    if method == "universal":
        cumulative = np.cumsum(_fitness_probabilities(errors))
        pointers = rng.random() / amount + np.arange(amount) / amount
        return np.clip(np.searchsorted(cumulative, pointers), 0, population_size - 1)

    if method == "boltzmann":
        progress = generation / max(1, max_generations)
        temperature = max(
            boltzmann_min_temperature,
            boltzmann_temperature * (1.0 - progress),
        )
        values = np.exp(-(errors - errors.min()) / temperature)
        return rng.choice(population_size, amount, p=values / values.sum())

    if method == "ranking":
        if not 1 <= ranking_pressure <= 2:
            raise ValueError("ranking_pressure debe estar entre 1 y 2")
        ordered = np.argsort(errors)
        if population_size == 1:
            return np.zeros(amount, dtype=int)
        ranks = np.arange(population_size, 0, -1)
        probabilities = (
            (2 - ranking_pressure) / population_size
            + 2
            * (ranks - 1)
            * (ranking_pressure - 1)
            / (population_size * (population_size - 1))
        )
        return rng.choice(ordered, amount, p=probabilities / probabilities.sum())

    size = min(tournament_size, population_size)
    selected = np.empty(amount, dtype=int)
    for position in range(amount):
        candidates = rng.choice(population_size, size, replace=False)
        candidates = candidates[np.argsort(errors[candidates])]
        if method == "tournament_deterministic":
            selected[position] = candidates[0]
            continue
        for candidate_position, candidate in enumerate(candidates):
            if candidate_position == size - 1 or rng.random() < tournament_probability:
                selected[position] = candidate
                break
    return selected


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    triangle_count: int,
    method: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruza por triángulos completos para no cortar un genotipo RGBA/geométrico."""
    if method not in CROSSOVER_METHODS:
        raise ValueError(f"Cruza desconocida: {method}")
    blocks_a = parent_a.reshape(triangle_count, 10)
    blocks_b = parent_b.reshape(triangle_count, 10)

    if triangle_count < 2:
        return parent_a.copy(), parent_b.copy()

    if method == "one_point":
        point = int(rng.integers(1, triangle_count))
        child_a = np.concatenate((blocks_a[:point], blocks_b[point:]))
        child_b = np.concatenate((blocks_b[:point], blocks_a[point:]))
    elif method == "two_point" and triangle_count >= 3:
        point_a, point_b = np.sort(rng.choice(np.arange(1, triangle_count), 2, replace=False))
        child_a = blocks_a.copy()
        child_b = blocks_b.copy()
        child_a[point_a:point_b] = blocks_b[point_a:point_b]
        child_b[point_a:point_b] = blocks_a[point_a:point_b]
    elif method == "uniform":
        mask = rng.random(triangle_count) < 0.5
        child_a = np.where(mask[:, None], blocks_a, blocks_b)
        child_b = np.where(mask[:, None], blocks_b, blocks_a)
    elif method == "region":
        # Intercambia una región espacial coherente, no cientos de capas inconexas.
        axis = int(rng.integers(0, 2))
        cut = rng.uniform(0.25, 0.75)
        centroids = (
            blocks_a[:, axis:6:2].mean(axis=1)
            + blocks_b[:, axis:6:2].mean(axis=1)
        ) / 2
        mask = centroids < cut
        if rng.random() < 0.5:
            mask = ~mask
        child_a = np.where(mask[:, None], blocks_a, blocks_b)
        child_b = np.where(mask[:, None], blocks_b, blocks_a)
    else:
        # Con dos triángulos, una cruza de dos puntos degenera en un punto.
        point = 1
        child_a = np.concatenate((blocks_a[:point], blocks_b[point:]))
        child_b = np.concatenate((blocks_b[:point], blocks_a[point:]))

    return child_a.reshape(-1).copy(), child_b.reshape(-1).copy()


def mutate(
    individual: np.ndarray,
    method: str,
    probability: float,
    generation: int,
    max_generations: int,
    rng: np.random.Generator,
    *,
    max_genes: int = 12,
    guide_points: np.ndarray | None = None,
    guide_colors: np.ndarray | None = None,
) -> np.ndarray:
    """Mutaciones Gen, MultiGen y No Uniforme, seleccionables por nombre."""
    if method not in MUTATION_METHODS:
        raise ValueError(f"Mutación desconocida: {method}")
    child = individual.copy()
    if rng.random() >= probability:
        return child

    if method == "triangle":
        blocks = child.reshape(-1, 10)
        progress = generation / max(1, max_generations)
        # Siempre queda una mezcla de pasos finos y medianos: no se congela al final.
        sigma = 0.06 * (1.0 - progress) + 0.008
        amount = int(rng.integers(1, min(4, len(blocks)) + 1))
        for triangle_index in rng.choice(len(blocks), amount, replace=False):
            triangle = blocks[triangle_index]
            points = triangle[:6].reshape(3, 2)
            operation = rng.choice(
                ("translate", "scale", "vertex", "color", "alpha", "reset"),
                p=(0.22, 0.18, 0.18, 0.18, 0.08, 0.16),
            )
            if operation == "translate":
                points += rng.normal(0.0, sigma, size=2)
            elif operation == "scale":
                center = points.mean(axis=0)
                factor = np.exp(rng.normal(0.0, 2.0 * sigma))
                points[:] = center + (points - center) * factor
            elif operation == "vertex":
                points[int(rng.integers(0, 3))] += rng.normal(0.0, sigma, size=2)
            elif operation == "color":
                triangle[6:9] += rng.normal(0.0, 1.5 * sigma, size=3)
            elif operation == "alpha":
                triangle[9] += rng.normal(0.0, 1.5 * sigma)
            else:
                guided = guide_points is not None and len(guide_points) and rng.random() < 0.8
                guide_index = int(rng.integers(0, len(guide_points))) if guided else None
                center = guide_points[guide_index] if guided else rng.random(2)
                radius = np.clip(rng.lognormal(mean=-3.0, sigma=0.65), 0.008, 0.22)
                angles = rng.uniform(0.0, 2.0 * np.pi, size=3)
                points[:] = center + np.column_stack((np.cos(angles), np.sin(angles))) * radius
                if guided and guide_colors is not None:
                    triangle[6:9] = np.clip(
                        guide_colors[guide_index] + rng.normal(0.0, 0.04, size=3), 0.0, 1.0
                    )
                else:
                    triangle[6:9] = rng.random(3)
                triangle[9] = rng.uniform(0.35, 0.9)
            triangle[:] = np.clip(triangle, 0.0, 1.0)
            if operation == "reset" and len(blocks) > 1:
                # Los reemplazos pequeños deben quedar sobre las masas globales.
                top_start = max(0, int(len(blocks) * 0.8))
                top_index = int(rng.integers(top_start, len(blocks)))
                blocks[[triangle_index, top_index]] = blocks[[top_index, triangle_index]]
        # Alterar el orden permite que un buen triángulo deje de quedar oculto.
        if len(blocks) > 1 and rng.random() < 0.08:
            first, second = rng.choice(len(blocks), 2, replace=False)
            blocks[[first, second]] = blocks[[second, first]]
        return child

    if method == "gene":
        indices = np.asarray([rng.integers(0, len(child))])
        deltas = rng.uniform(-0.15, 0.15, 1)
    elif method == "multigene":
        amount = int(rng.integers(1, min(max_genes, len(child)) + 1))
        indices = rng.choice(len(child), amount, replace=False)
        deltas = rng.normal(0.0, 0.08, amount)
    else:
        # Mutación no uniforme: los cambios disminuyen al avanzar las generaciones.
        amount = int(rng.integers(1, min(max_genes, len(child)) + 1))
        indices = rng.choice(len(child), amount, replace=False)
        progress = generation / max(1, max_generations)
        sigma = 0.15 * (1.0 - progress) + 0.01
        deltas = rng.normal(0.0, sigma, amount)

    child[indices] = np.clip(child[indices] + deltas, 0.0, 1.0)
    return child


def choose_survivors(
    parents: Sequence[np.ndarray],
    parent_fitness: Sequence[float],
    children: Sequence[np.ndarray],
    child_fitness: Sequence[float],
    amount: int,
    method: str,
) -> tuple[list[np.ndarray], list[float]]:
    """Supervivencia aditiva (padres+hijos) o exclusiva (hijos)."""
    if method not in SURVIVAL_METHODS:
        raise ValueError(f"Supervivencia desconocida: {method}")

    if method == "additive":
        pool = list(parents) + list(children)
        fitness = list(parent_fitness) + list(child_fitness)
    else:
        pool = list(children)
        fitness = list(child_fitness)
        if len(pool) < amount:
            ordered_parents = np.argsort(parent_fitness)
            missing = amount - len(pool)
            pool.extend(parents[index] for index in ordered_parents[:missing])
            fitness.extend(parent_fitness[index] for index in ordered_parents[:missing])

    selected = np.argsort(fitness)[:amount]
    return [pool[index] for index in selected], [float(fitness[index]) for index in selected]
