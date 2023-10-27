from graphs import createRandomGraph, createFullGraph, setGraphSeed, drawGraph
from evolution import generatePopulation, rankSpecimens, runTournaments, runMutations
from time import process_time

POPULATION_COUNT = 1000
MUTATION_PROBABILITY = 0.5
BIT_CHANGE_PROBABILITY = 0.2
MAX_ITERATIONS = 50


graph = createRandomGraph(50, 0.1)
population0 = generatePopulation(POPULATION_COUNT)

iterations = 0
population1 = rankSpecimens(graph, population0)
for specimen in population1:
    print(f'{specimen.score}, {specimen.rank}, {specimen.genotype_as_string()}')
while iterations < MAX_ITERATIONS:
    start = process_time()
    print(iterations)
    population2 = runTournaments(population1)
    print('after tournaments:')
    for specimen in population2:
        print(f'{specimen.score}, {specimen.rank}, {specimen.genotype_as_string()}')
    population3 = runMutations(MUTATION_PROBABILITY,BIT_CHANGE_PROBABILITY, population2)
    print('after mutations:')
    population4 = rankSpecimens(graph, population3)
    for specimen in population4:
        print(f'{specimen.score}, {specimen.rank}, {specimen.genotype_as_string()}')
    iterations += 1
    end = process_time()
    print(population4[0].genotype_as_string())
    print(end-start)
    population1 = population4

setGraphSeed(graph, population1[0].genotype)
print(population1[0].genotype)
drawGraph(graph)
