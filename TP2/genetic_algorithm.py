import csv
import json
import random

import numpy as np

from individual import Individual
from operators import (
    tournament_selection,
    one_point_crossover,
    mutate_gene,
)


class GeneticAlgorithm:
    def __init__(
            self,
            target_image,
            num_triangles,
            population_size=100,
            generations=500,
            mutation_probability=0.3,
            crossover_probability=0.9,
            tournament_size=3,
            elitism=1
    ):

        self.target_image = target_image.convert("RGB")
        self.width, self.height = self.target_image.size
        self.target_array = np.asarray(
            self.target_image,
            dtype=np.float32
        )
        self.num_triangles = num_triangles

        self.population_size = population_size
        self.generations = generations

        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability

        self.tournament_size = tournament_size

        self.elitism = elitism

        self.population = []

        self.history = []

    def initialize_population(self):
        self.population = [
            Individual.random(
                num_triangles=self.num_triangles,
                width=self.width,
                height=self.height
            )
            for _ in range(self.population_size)
        ]

    def evaluate_population(self):
        for individual in self.population:
            individual.evaluate(
                self.target_array,
                self.width,
                self.height
            )

    def sort_population(self):
        self.population.sort(
            key=lambda x: x.fitness,
            reverse=True
        )

    def create_new_generation(self):
        new_population = []

        # ELITISMO
        # copiamos directamente los mejores individuos
        for i in range(self.elitism):
            elite = self.population[i].clone()

            new_population.append(elite)

        while len(new_population) < self.population_size:
            parent1 = tournament_selection(
                self.population,
                self.tournament_size
            )

            parent2 = tournament_selection(
                self.population,
                self.tournament_size
            )

            # CRUZA
            if random.random() < self.crossover_probability:
                child1, child2 = one_point_crossover(
                    parent1,
                    parent2
                )

            else:
                child1 = parent1.clone()
                child2 = parent2.clone()

            # MUTACION
            if random.random() < self.mutation_probability:
                mutate_gene(
                    child1,
                    self.width,
                    self.height
                )

            if random.random() < self.mutation_probability:
                mutate_gene(
                    child2,
                    self.width,
                    self.height
                )

            new_population.append(child1)

            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population

    def run(self):

        self.initialize_population()

        best_individual = None

        for generation in range(self.generations):
            self.evaluate_population()

            self.sort_population()

            best_individual = self.population[0]

            average_fitness = np.mean([
                individual.fitness
                for individual in self.population
            ])

            average_error = np.mean([
                individual.error
                for individual in self.population
            ])

            self.history.append({
                "generation": generation,
                "best_fitness": best_individual.fitness,
                "best_error": best_individual.error,
                "average_fitness": average_fitness,
                "average_error": average_error,
            })

            print(
                f"Generacion {generation:4d} | "
                f"Best error: {best_individual.error:10.2f} | "
                f"Fitness: {best_individual.fitness:.8f}"
            )

            # Guardamos imagenes intermedias
            if generation % 50 == 0:

                image = best_individual.render(
                    self.width,
                    self.height
                )

                image.save(
                    f"generation_{generation:04d}.png"
                )

            if generation < self.generations - 1:
                self.create_new_generation()

        return best_individual

    def save_history(self, filename="history.csv"):

        if not self.history:
            return

        with open(filename, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=self.history[0].keys()
            )

            writer.writeheader()
            writer.writerows(self.history)

    def save_triangles(
            self,
            individual,
            filename="best_triangles.json"
    ):
        triangles = [
            triangle.to_dict()
            for triangle in individual.triangles
        ]

        with open(filename, "w") as file:
            json.dump(
                triangles,
                file,
                indent=4
            )
