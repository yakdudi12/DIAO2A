import random

from individual import Individual


def tournament_selection(population, tournament_size=3):
    """
    Selección por torneo determinista

    Se toman K individuos aleatorios y se devuelve
    el que tenga mayor fitness
    """

    competitors = random.sample(
        population,
        tournament_size
    )

    winner = max(
        competitors,
        key=lambda individual: individual.fitness
    )

    return winner

def one_point_crossover(parent1, parent2):
    """
    Cruce de un punto

    El punto de corte se realiza entre triángulos,
    no entre los parámetros internos de cada triángulo
    """

    num_triangles = len(parent1.triangles)

    if num_triangles < 2:
        return parent1.clone(), parent2.clone()

    crossover_point = random.randint(
        1,
        num_triangles - 1
    )

    child1_triangles = (
        parent1.triangles[:crossover_point]
        +
        parent2.triangles[crossover_point:]
    )

    child2_triangles = (
        parent2.triangles[:crossover_point]
        +
        parent1.triangles[crossover_point:]
    )

    child1 = Individual(child1_triangles)
    child2 = Individual(child2_triangles)

    # Para evitar que padres e hijos compartan objetos Triangle
    child1 = child1.clone()
    child2 = child2.clone()

    return child1, child2

def mutate_gene(individual, width, height):
    """
    Mutación de unico gen

    1. Elegimos un triángulo
    2. Elegimos uno de sus parámetros
    3. Lo modificamos
    """

    triangle = random.choice(individual.triangles)

    gene = random.choice([
        "x1", "y1",
        "x2", "y2",
        "x3", "y3",
        "r", "g", "b", "a"
    ])

    # Coordenada X
    if gene in ("x1", "x2", "x3"):
        current_value = getattr(triangle, gene)
        mutation = random.randint(-10, 10)
        new_value = current_value + mutation
        new_value = max(
            0,
            min(width - 1, new_value)
        )

    # Coordenada Y
    elif gene in ("y1", "y2", "y3"):
        current_value = getattr(triangle, gene)
        mutation = random.randint(-10, 10)
        new_value = current_value + mutation
        new_value = max(
            0,
            min(height - 1, new_value)
        )

    # Color / alpha
    else:
        current_value = getattr(triangle, gene)
        mutation = random.randint(-25, 25)
        new_value = current_value + mutation
        new_value = max(
            0,
            min(255, new_value)
        )

    setattr(
        triangle,
        gene,
        new_value
    )