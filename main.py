from graphs import createRandomGraph, createFullGraph, setGraphSeed, drawGraph
from evolution import generatePopulation, rankSpecimens, runTournaments, runMutations
from time import process_time

POPULATION_COUNT = 100
MUTATION_PROBABILITY = 0.2
BIT_CHANGE_PROBABILITY = 0.05
MAX_ITERATIONS = 5


graph = createRandomGraph(50, 0.1)
population = generatePopulation(POPULATION_COUNT)

iterations = 0
rankSpecimens(graph, population)
for specimen in population:
    print(f'{specimen.score}, {specimen.rank}, {specimen.genotype_as_string()}')
while iterations < MAX_ITERATIONS:
    start = process_time()
    print(iterations)
    population = runTournaments(population)
    runMutations(BIT_CHANGE_PROBABILITY, MUTATION_PROBABILITY, population)
    rankSpecimens(graph, population)
    for specimen in population:
        print(f'{specimen.score}, {specimen.rank}, {specimen.genotype_as_string()}')
    population = population
    iterations += 1
    end = process_time()
    print(population[0].genotype_as_string())
    print(end-start)

setGraphSeed(graph, population[0].genotype)
print(population[0].genotype)
drawGraph(graph)
