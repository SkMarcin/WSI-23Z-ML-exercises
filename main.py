from inspect import _void
from graphs import createRandomGraph, createFullGraph, setGraphSeed, drawGraph
from evolution import generatePopulation, rankSpecimens, runTournaments, runMutations
from time import process_time
import matplotlib.pyplot as plt

POPULATION_COUNT = 500
MUTATION_PROBABILITY = 0.5
BIT_CHANGE_PROBABILITY = 0.02
MAX_ITERATIONS = 250

gen_scores = []
it = []

graph = createRandomGraph(10, 0.5)
population = generatePopulation(POPULATION_COUNT)
lowest_score = 10000
best_specimen = 1
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
    if population[0].score < lowest_score:
        lowest_score = population[0].score
        best_specimen = population[0]
    it.append(iterations)
    gen_scores.append(population[0].score)


# setGraphSeed(graph, population[0].genotype)
# print(population[0].genotype)
print(lowest_score, "lowest")
setGraphSeed(graph, best_specimen.genotype)
drawGraph(graph)


plt.plot(it, gen_scores)
plt.show()