import random

def tournament_selection(population, tournament_size=3):
    participants = random.sample(population, tournament_size)

    return max(
        participants,
        
    )