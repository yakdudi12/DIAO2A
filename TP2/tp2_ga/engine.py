"""Motor principal del algoritmo genético."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .fitness import FitnessEvaluator, gradient
from .operators import (
    CROSSOVER_METHODS,
    MUTATION_METHODS,
    SELECTION_METHODS,
    SURVIVAL_METHODS,
    choose_survivors,
    crossover,
    mutate,
    select_indices,
)


@dataclass(slots=True)
class GAConfig:
    """Hiperparámetros del algoritmo genético."""

    population_size: int = 50
    generations: int = 5_000
    mutation_probability: float = 0.20
    crossover_probability: float = 0.65
    elite_size: int = 5
    random_injection: int = 2
    stop_error: float = 0.001
    patience: int = 500
    evaluation_size: int = 64
    initial_evaluation_size: int | None = 48
    seed: int | None = 42
    use_gpu: bool = True

    selection: str = "ranking"
    crossover: str = "region"
    mutation: str = "triangle"
    survival: str = "exclusive"

    tournament_size: int = 3
    tournament_probability: float = 0.8
    ranking_pressure: float = 1.5
    boltzmann_temperature: float = 1.0
    boltzmann_min_temperature: float = 0.1

    color_weight: float = 0.55
    edge_weight: float = 0.30
    ssim_weight: float = 0.15
    detail_boost: float = 4.0
    pyramid_scales: tuple[float, ...] = (0.5, 1.0)
    pyramid_weights: tuple[float, ...] = (0.25, 0.75)
    resolution_change_fractions: tuple[float, float] = (0.35, 0.65)
    target_guided_initialization: bool = False
    progress: bool = True

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size debe ser al menos 2")
        if self.generations <= 0:
            raise ValueError("generations debe ser positivo")
        if not 0 <= self.mutation_probability <= 1:
            raise ValueError("mutation_probability debe estar en [0, 1]")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability debe estar en [0, 1]")
        if not 0 <= self.elite_size < self.population_size:
            raise ValueError("elite_size debe estar entre 0 y population_size-1")
        available = self.population_size - self.elite_size
        if not 0 <= self.random_injection < available:
            raise ValueError("random_injection debe dejar lugar para descendencia")
        if self.patience <= 0 or self.evaluation_size < 7:
            raise ValueError("patience debe ser positivo y evaluation_size al menos 7")
        if self.stop_error < 0:
            raise ValueError("stop_error no puede ser negativo")
        if self.initial_evaluation_size is not None and not (
            7 <= self.initial_evaluation_size <= self.evaluation_size
        ):
            raise ValueError("initial_evaluation_size debe estar entre 7 y evaluation_size")
        if self.selection not in SELECTION_METHODS:
            raise ValueError(f"Selección desconocida: {self.selection}")
        if self.crossover not in CROSSOVER_METHODS:
            raise ValueError(f"Cruza desconocida: {self.crossover}")
        if self.mutation not in MUTATION_METHODS:
            raise ValueError(f"Mutación desconocida: {self.mutation}")
        if self.survival not in SURVIVAL_METHODS:
            raise ValueError(f"Supervivencia desconocida: {self.survival}")
        if self.tournament_size <= 0 or not 0 <= self.tournament_probability <= 1:
            raise ValueError("Parámetros de torneo inválidos")
        if not 1 <= self.ranking_pressure <= 2:
            raise ValueError("ranking_pressure debe estar entre 1 y 2")
        if self.color_weight < 0 or self.edge_weight < 0 or self.ssim_weight < 0:
            raise ValueError("Los pesos del fitness no pueden ser negativos")
        if self.color_weight + self.edge_weight + self.ssim_weight <= 0:
            raise ValueError("Al menos un peso del fitness debe ser positivo")
        if self.detail_boost < 0:
            raise ValueError("detail_boost no puede ser negativo")
        if (
            not self.pyramid_scales
            or len(self.pyramid_scales) != len(self.pyramid_weights)
            or min(self.pyramid_scales) <= 0
            or min(self.pyramid_weights) < 0
            or sum(self.pyramid_weights) <= 0
        ):
            raise ValueError("La pirámide de fitness es inválida")
        first_change, final_change = self.resolution_change_fractions
        if not 0 < first_change < final_change < 1:
            raise ValueError("Las fracciones de resolución deben ser crecientes y estar en (0, 1)")


@dataclass(slots=True)
class RunResult:
    best_genes: np.ndarray
    best_fitness: float
    history: list[float]
    resolution_history: list[int]
    diversity_history: list[int]
    snapshots: list[tuple[int, float, np.ndarray]]
    generations_completed: int
    elapsed_seconds: float
    stop_reason: str
    backend: str
    config: GAConfig = field(repr=False)


class GeneticImageGA:
    """AG configurable cuyo individuo contiene `K` triángulos RGBA."""

    genes_per_triangle = 10

    def __init__(self, target: Image.Image, triangle_count: int, config: GAConfig | None = None):
        if triangle_count <= 0:
            raise ValueError("triangle_count debe ser positivo")
        self.target = target.convert("RGB")
        self.triangle_count = int(triangle_count)
        self.gene_count = self.triangle_count * self.genes_per_triangle
        self.config = config or GAConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.evaluator = FitnessEvaluator(
            self.target,
            self.triangle_count,
            use_gpu=self.config.use_gpu,
            color_weight=self.config.color_weight,
            edge_weight=self.config.edge_weight,
            ssim_weight=self.config.ssim_weight,
            detail_boost=self.config.detail_boost,
            pyramid_scales=self.config.pyramid_scales,
            pyramid_weights=self.config.pyramid_weights,
        )
        self.population: list[np.ndarray] = []
        self.fitness: list[float] = []
        self.best_genes: np.ndarray | None = None
        self.best_fitness = float("inf")

    @property
    def initial_size(self) -> int:
        return self.config.initial_evaluation_size or self.config.evaluation_size

    def _target_guides(self, amount: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Muestra centros preferentemente en bordes, conservando exploración global."""
        rgb = np.asarray(self.target, dtype=np.float64) / 255.0
        gray = np.asarray(self.target.convert("L"), dtype=np.float64) / 255.0
        detail = gradient(gray)
        probabilities = (0.15 + detail / max(float(detail.max()), 1e-8)).ravel()
        probabilities /= probabilities.sum()
        indices = self.rng.choice(rgb.shape[0] * rgb.shape[1], amount, p=probabilities)
        rows, columns = np.unravel_index(indices, gray.shape)
        points = np.column_stack((columns / max(1, rgb.shape[1] - 1), rows / max(1, rgb.shape[0] - 1)))
        return points, rgb[rows, columns]

    def _new_individual(self) -> np.ndarray:
        if not self.config.target_guided_initialization:
            return self.rng.random(self.gene_count)
        blocks = np.empty((self.triangle_count, self.genes_per_triangle), dtype=np.float64)
        guide_indices = self.rng.integers(0, len(self.guide_points), size=self.triangle_count)
        centers = self.guide_points[guide_indices]
        # Una minoría de triángulos globales modela masas; el resto queda disponible para detalle.
        global_mask = self.rng.random(self.triangle_count) < 0.20
        radii = np.clip(self.rng.lognormal(-2.6, 0.75, self.triangle_count), 0.008, 0.28)
        radii[global_mask] = self.rng.uniform(0.22, 0.65, global_mask.sum())
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=(self.triangle_count, 3))
        points = centers[:, None, :] + np.stack((np.cos(angles), np.sin(angles)), axis=2) * radii[:, None, None]
        blocks[:, :6] = np.clip(points, 0.0, 1.0).reshape(self.triangle_count, 6)
        blocks[:, 6:9] = np.clip(
            self.guide_colors[guide_indices] + self.rng.normal(0.0, 0.08, size=(self.triangle_count, 3)),
            0.0,
            1.0,
        )
        blocks[:, 9] = self.rng.uniform(0.18, 0.72, self.triangle_count)
        # El orden es parte del fenotipo: masas grandes abajo, detalle fino arriba.
        return blocks[np.argsort(radii)[::-1]].reshape(-1)

    def initialize_random_population(self) -> None:
        """Inicializa la población usando exclusivamente el generador de la corrida."""
        if self.config.target_guided_initialization:
            self.guide_points, self.guide_colors = self._target_guides()
        else:
            # No se consulta el objetivo ni se consume RNG antes de crear la población.
            # Así, para una seed y dimensiones dadas, la generación 0 es idéntica
            # independientemente de la imagen que se quiera aproximar.
            self.guide_points = None
            self.guide_colors = None
        self.population = [self._new_individual() for _ in range(self.config.population_size)]

    def genes_to_triangles(self, genes: np.ndarray):
        blocks = np.asarray(genes).reshape(self.triangle_count, self.genes_per_triangle)
        return [(block[:6].copy(), block[6:].copy()) for block in blocks]

    def _evaluate_population(self, size: int) -> None:
        self.fitness = self.evaluator.evaluate_many(self.population, size)
        self._update_best(reset=True)

    def _update_best(self, *, reset: bool = False) -> None:
        if reset:
            self.best_fitness = float("inf")
            self.best_genes = None
        index = int(np.argmin(self.fitness))
        if self.fitness[index] < self.best_fitness:
            self.best_fitness = float(self.fitness[index])
            self.best_genes = self.population[index].copy()

    def _resolution_schedule(self) -> dict[int, int]:
        if self.initial_size >= self.config.evaluation_size:
            return {}
        intermediate = round((self.initial_size + self.config.evaluation_size) / 2)
        first_change, final_change = self.config.resolution_change_fractions
        return {
            int(self.config.generations * first_change): intermediate,
            int(self.config.generations * final_change): self.config.evaluation_size,
        }

    def _phenotype_diversity(self, size: int) -> int:
        """Diversidad tras cuantizar como el rasterizador, no floats casi equivalentes."""
        signatures = set()
        for individual in self.population:
            blocks = individual.reshape(self.triangle_count, self.genes_per_triangle)
            quantized = np.empty_like(blocks, dtype=np.int16)
            quantized[:, 0:6:2] = np.rint(blocks[:, 0:6:2] * (size - 1))
            quantized[:, 1:6:2] = np.rint(blocks[:, 1:6:2] * (size - 1))
            quantized[:, 6:] = np.rint(blocks[:, 6:] * 255)
            signatures.add(quantized.tobytes())
        return len(signatures)

    def _selection_indices(self, amount: int, generation: int) -> np.ndarray:
        return select_indices(
            self.fitness,
            amount,
            self.config.selection,
            self.rng,
            generation=generation,
            max_generations=self.config.generations,
            tournament_size=self.config.tournament_size,
            tournament_probability=self.config.tournament_probability,
            ranking_pressure=self.config.ranking_pressure,
            boltzmann_temperature=self.config.boltzmann_temperature,
            boltzmann_min_temperature=self.config.boltzmann_min_temperature,
        )

    def _create_children(self, generation: int, amount: int) -> list[np.ndarray]:
        parent_count = max(2, amount + amount % 2)
        parent_indices = self._selection_indices(parent_count, generation)
        parents = [self.population[index] for index in parent_indices]
        children: list[np.ndarray] = []
        while len(children) < amount:
            first, second = self.rng.choice(len(parents), 2, replace=False)
            parent_a, parent_b = parents[first], parents[second]
            if self.rng.random() < self.config.crossover_probability:
                child_a, child_b = crossover(
                    parent_a,
                    parent_b,
                    self.triangle_count,
                    self.config.crossover,
                    self.rng,
                )
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
            children.append(
                mutate(
                    child_a,
                    self.config.mutation,
                    self.config.mutation_probability,
                    generation,
                    self.config.generations,
                    self.rng,
                    guide_points=self.guide_points,
                    guide_colors=self.guide_colors,
                )
            )
            if len(children) < amount:
                children.append(
                    mutate(
                        child_b,
                        self.config.mutation,
                        self.config.mutation_probability,
                        generation,
                        self.config.generations,
                        self.rng,
                        guide_points=self.guide_points,
                        guide_colors=self.guide_colors,
                    )
                )
        return children

    def run(
        self,
        *,
        progress_desc: str = "Evolución AG",
        progress_position: int = 0,
        progress_leave: bool = True,
        progress_callback: Callable[[int, float, int, int], None] | None = None,
    ) -> RunResult:
        """Ejecuta una corrida independiente y devuelve métricas estructuradas."""
        start = time.perf_counter()
        self.rng = np.random.default_rng(self.config.seed)
        self.initialize_random_population()
        current_size = self.initial_size
        self._evaluate_population(current_size)

        history = [self.best_fitness]
        resolution_history = [current_size]
        diversity_history = [self._phenotype_diversity(current_size)]
        snapshots = [(0, self.best_fitness, self.best_genes.copy())]
        snapshot_generations = {
            round(self.config.generations * fraction)
            for fraction in (0.25, 0.50, 0.75, 1.0)
        }
        schedule = self._resolution_schedule()
        stagnation = 0
        previous_best = self.best_fitness
        stop_reason = "maximum_generations"
        generations_completed = 0

        progress = range(self.config.generations)
        if self.config.progress and progress_callback is None:
            progress = tqdm(
                progress,
                desc=progress_desc,
                unit="gen",
                dynamic_ncols=True,
                position=progress_position,
                leave=progress_leave,
            )
        for generation in progress:
            if generation in schedule and schedule[generation] != current_size:
                current_size = schedule[generation]
                self._evaluate_population(current_size)
                previous_best = self.best_fitness
                stagnation = 0
                if self.config.progress and progress_callback is None:
                    progress.write(f"Evaluación aumentada a lado {current_size}")

            elite_indices = np.argsort(self.fitness)[: self.config.elite_size]
            elites = [self.population[index].copy() for index in elite_indices]
            elite_fitness = [self.fitness[index] for index in elite_indices]
            elite_set = set(map(int, elite_indices))
            remaining_parents = [
                individual
                for index, individual in enumerate(self.population)
                if index not in elite_set
            ]
            remaining_parent_fitness = [
                value for index, value in enumerate(self.fitness) if index not in elite_set
            ]

            slots = self.config.population_size - self.config.elite_size
            evolved_children = self._create_children(
                generation, slots - self.config.random_injection
            )
            random_children = [self._new_individual() for _ in range(self.config.random_injection)]
            children = evolved_children + random_children
            child_fitness = self.evaluator.evaluate_many(children, current_size)
            survivors, survivor_fitness = choose_survivors(
                remaining_parents,
                remaining_parent_fitness,
                children,
                child_fitness,
                slots,
                self.config.survival,
            )
            self.population = elites + survivors
            self.fitness = elite_fitness + survivor_fitness
            self._update_best()

            generations_completed = generation + 1
            history.append(self.best_fitness)
            resolution_history.append(current_size)
            diversity_history.append(self._phenotype_diversity(current_size))
            if generations_completed in snapshot_generations:
                snapshots.append(
                    (generations_completed, self.best_fitness, self.best_genes.copy())
                )

            if self.best_fitness < previous_best - 1e-6:
                previous_best = self.best_fitness
                stagnation = 0
            else:
                stagnation += 1

            if (
                self.config.progress
                and progress_callback is None
                and (generation % 10 == 0 or generations_completed == self.config.generations)
            ):
                progress.set_postfix(
                    error=f"{self.best_fitness:.6f}",
                    res=current_size,
                    sin_mejora=stagnation,
                )

            if progress_callback is not None:
                progress_callback(
                    generations_completed,
                    self.best_fitness,
                    current_size,
                    stagnation,
                )

            final_resolution = current_size == self.config.evaluation_size
            if final_resolution and self.best_fitness < self.config.stop_error:
                stop_reason = "target_error"
                break
            if final_resolution and stagnation >= self.config.patience:
                stop_reason = "stagnation"
                break

        # CUDA y Pillow difieren levemente en las reglas de borde/alpha. Como la imagen
        # exportada usa Pillow, se reordena una vez la población final con ese renderer.
        final_fitness = self.evaluator.evaluate_many_cpu(
            self.population, self.config.evaluation_size
        )
        final_index = int(np.argmin(final_fitness))
        self.best_fitness = float(final_fitness[final_index])
        self.best_genes = self.population[final_index].copy()
        history[-1] = self.best_fitness

        if not snapshots or snapshots[-1][0] != generations_completed:
            snapshots.append(
                (generations_completed, self.best_fitness, self.best_genes.copy())
            )
        else:
            snapshots[-1] = (
                generations_completed, self.best_fitness, self.best_genes.copy()
            )
        elapsed = time.perf_counter() - start
        return RunResult(
            best_genes=self.best_genes.copy(),
            best_fitness=self.best_fitness,
            history=history,
            resolution_history=resolution_history,
            diversity_history=diversity_history,
            snapshots=snapshots,
            generations_completed=generations_completed,
            elapsed_seconds=elapsed,
            stop_reason=stop_reason,
            backend=self.evaluator.backend,
            config=self.config,
        )
