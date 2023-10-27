from graphs import createRandomGraph, createFullGraph, setGraphSeed, drawGraph
from evolution import generatePopulation, rankSpecimens, runTournaments, runMutations
from time import process_time

POPULATION_COUNT = 1000
MUTATION_PROBABILITY = 0.5
BIT_CHANGE_PROBABILITY = 0.02
MAX_ITERATIONS = 50


graph = createRandomGraph(50, 0.1)
population = generatePopulation(POPULATION_COUNT)

iterations = 0
population = rankSpecimens(graph, population)
while iterations < MAX_ITERATIONS:
    start = process_time()
    print(iterations)
    population = runTournaments(population)
    population = runMutations(MUTATION_PROBABILITY,BIT_CHANGE_PROBABILITY, population)
    population = rankSpecimens(graph, population)
    iterations += 1
    end = process_time()
    print(end-start)
    population = population
    print(population[0].genotype_as_string())
    print(population[0].score)

setGraphSeed(graph, population[0].genotype)
print(population[0].genotype)
drawGraph(graph)
