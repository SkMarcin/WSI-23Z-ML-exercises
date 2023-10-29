from inspect import _void
from graphs import createRandomGraph, createFullGraph, setGraphSeed, drawGraph, mean, standard_deviation
from evolution import generatePopulation, rankSpecimens, runTournaments, runMutations
from time import process_time
import matplotlib.pyplot as plt

POPULATION_COUNT = 100
MUTATION_PROBABILITY = 0.25
BIT_CHANGE_PROBABILITY = 0.1
MAX_ITERATIONS = 5
NODES = 50
EDGE_PROBABILITY = 0.3
ATTEMPTS = 10

running = True
graph = createRandomGraph(NODES, EDGE_PROBABILITY)
drawGraph(graph)
while running:
    print(f'\nMutation probability: {MUTATION_PROBABILITY}')
    print(f'Bit change probability: {BIT_CHANGE_PROBABILITY}')
    print(f'Nodes in graph: {NODES}')

    temp = int(input("\nDo you want to change any of these parameters?\n1 if yes, 0 if no: "))
    loop_running = True
    while(loop_running):
        if temp == 1:
            MUTATION_PROBABILITY = float(input("Enter mutation probability: "))
            BIT_CHANGE_PROBABILITY = float(input("Enter bit change probability: "))
            temp2 = int(input("Do you want to generate new graph?\n1 if yes, 0 if no: "))
            while(loop_running):
                if temp2 == 1:
                    NODES = int(input("How many nodes for graph: "))
                    EDGE_PROBABILITY = float(input("How many edges to be in graph (probability between 0 and 1): "))
                    graph = createRandomGraph(NODES, EDGE_PROBABILITY)
                    loop_running = False
                elif temp2 == 0:
                    loop_running = False
                else:
                    temp2 = int(input("enter valid value: "))

        elif temp == 0:
            loop_running = False
        else:
            temp = int(input("enter valid value: "))

    gen_scores = []
    it = []
    scores = []
    times = []
    graph = createRandomGraph(NODES, 0.3)

    POPULATION_COUNT = int(input("How large do you want the population: "))
    MAX_ITERATIONS = int(input("How many iterations you want to run: "))
    ATTEMPTS = int(input("How many attempts for average: "))


    for _ in range(0, ATTEMPTS):
        population = generatePopulation(POPULATION_COUNT, NODES)
        lowest_score = 1e20
        best_specimen = 1
        iterations = 0
        population = rankSpecimens(graph, population)
        start = process_time()
        while iterations < MAX_ITERATIONS:
            population = runTournaments(population)
            population = runMutations(MUTATION_PROBABILITY,BIT_CHANGE_PROBABILITY, population)
            population = rankSpecimens(graph, population)
            population = population
            it.append(iterations)
            gen_scores.append(population[0].score)
            iterations += 1
        end = process_time()
        times.append(end - start)
        scores.append(population[0].score)

    if ATTEMPTS == 1:
        setGraphSeed(graph, population[0].genotype)
        drawGraph(graph)
        plt.plot(it, gen_scores)
        plt.show()

    print(f'max: {max(scores)}')
    print(f'min: {min(scores)}')
    print(f'mean: {round(mean(scores), 2)}')
    print(f'deviation: {round(standard_deviation(scores), 2)}')
    print(f'average time per attempt: {round(mean(times), 2)}')

    temp = int(input("\nDo you want to continue?\n1 if yes, 0 if no: "))
    loop_running = True
    while(loop_running):
        if temp == 1:
            loop_running = False
        elif temp == 0:
            running = False
            loop_running = False
        else:
            temp = int(input("enter valid value: "))