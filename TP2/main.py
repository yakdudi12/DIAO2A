import argparse

from PIL import Image

from genetic_algorithm import GeneticAlgorithm


def main():
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm Image Compressor"
    )

    parser.add_argument(
        "image",
        type=str,
        help="Path to the target image"
    )

    parser.add_argument(
        "triangles",
        type=int,
        help="Amount of triangles"
    )

    parser.add_argument(
        "--population",
        type=int,
        default=100
    )

    parser.add_argument(
        "--generations",
        type=int,
        default=500
    )

    parser.add_argument(
        "--mutation",
        type=float,
        default=0.3
    )

    args = parser.parse_args()

    target_image = Image.open(args.image)

    ga = GeneticAlgorithm(
        target_image=target_image,
        num_triangles=args.triangles,
        population_size=args.population,
        generations=args.generations,
        mutation_probability=args.mutation,
        crossover_probability=0.9,
        tournament_size=3,
        elitism=1
    )

    best = ga.run()

    best_image = best.render(
        ga.width,
        ga.height
    )

    best_image.save("best.png")

    ga.save_history("history.csv")

    ga.save_triangles(
        best,
        "best_triangles.json"
    )

    print()
    print("Finished")
    print(f"Best error: {best.error}")
    print(f"Best fitness: {best.fitness}")
    print(f"Image saved to best.png")


if __name__ == "__main__":
    main()