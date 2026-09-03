import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tp2_ga import GAConfig, GeneticImageGA
from tp2_ga.fitness import FitnessEvaluator, combined_fitness
from tp2_ga.operators import crossover, mutate


class FitnessTests(unittest.TestCase):
    def test_identical_images_have_zero_error(self):
        image = Image.new("RGB", (16, 16), (30, 100, 220))
        self.assertAlmostEqual(combined_fitness(image, image), 0.0, places=6)

    def test_detail_difference_is_penalized(self):
        target = Image.new("RGB", (24, 24), "white")
        target.putpixel((12, 12), (0, 0, 0))
        reconstruction = Image.new("RGB", (24, 24), "white")
        self.assertGreater(combined_fitness(target, reconstruction), 0.0)


class OperatorTests(unittest.TestCase):
    def test_region_crossover_keeps_complete_triangles(self):
        rng = np.random.default_rng(2)
        parent_a = np.zeros(40)
        parent_b = np.ones(40)
        child, _ = crossover(parent_a, parent_b, 4, "region", rng)
        blocks = child.reshape(4, 10)
        self.assertTrue(all(np.all(block == 0) or np.all(block == 1) for block in blocks))

    def test_triangle_mutation_stays_in_bounds(self):
        rng = np.random.default_rng(3)
        individual = np.full(60, 0.5)
        child = mutate(individual, "triangle", 1.0, 20, 100, rng)
        self.assertTrue(np.all((0.0 <= child) & (child <= 1.0)))
        self.assertFalse(np.array_equal(individual, child))


class EngineTests(unittest.TestCase):
    def test_guided_initialization_has_local_triangles(self):
        target = Image.new("RGB", (24, 24), (120, 80, 40))
        config = GAConfig(
            population_size=4,
            generations=2,
            evaluation_size=16,
            initial_evaluation_size=None,
            elite_size=1,
            random_injection=1,
            use_gpu=False,
            progress=False,
        )
        ga = GeneticImageGA(target, 30, config)
        ga.initialize_random_population()
        self.assertEqual(len(ga.population), 4)
        self.assertTrue(all(individual.shape == (300,) for individual in ga.population))
        values = FitnessEvaluator(target, 30, use_gpu=False).evaluate_many(ga.population, 16)
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == "__main__":
    unittest.main()
