from graphs import (createFullGraph,
    createRandomGraph,
    drawGraph,
    setGraphSeed,
    transformToNXGraph,
    countLitUnlit,
)
from random import choices, random


class Specimen:
    def __init__(self, genotype_len=50, genotype=None):
        self.genotype_len = genotype_len
        self.genotype = genotype if genotype is not None else []
        if self.genotype == []:
            self.generate_genotype(genotype_len)
        self.score = -1
        self.rank = -1
        self.reproduction_probability = 0

    def generate_genotype(self, genotype_len):
        for i in range(0, genotype_len):
            self.genotype.append(choices([True, False])[0])

    def genotype_as_string(self):
        return ''.join('1' if bit else '0' for bit in self.genotype)


def generatePopulation(population_count, genotype_len):
    return [Specimen(genotype_len) for _ in range(population_count)]


def rankSpecimens(graph, specimens : list[Specimen]):
    for specimen in specimens:
        specimen.score = objectiveFunction(graph, specimen)
    specimens.sort(key=lambda x: x.score, reverse=True)
    i = 1
    for specimen in specimens:
        specimen.rank = i
        i += 1
    return specimens


def objectiveFunction(graph, specimen:Specimen):
    setGraphSeed(graph, specimen.genotype)
    G = transformToNXGraph(graph)
    lit, unlit = countLitUnlit(G)
    nodes_selected = specimen.genotype.count(True)
    cost = nodes_selected
    penalty = unlit * 100
    return cost + penalty


def runTournaments(specimens:list[Specimen]):
    n = len(list(specimens))
    new_specimens = []
    for _ in range (0, n):
        specimen1 = choices(specimens)[0]
        specimens.remove(specimen1)
        specimen2 = choices(specimens)[0]
        specimens.append(specimen1)
        if specimen1.rank < specimen2.rank:
            new_specimens.append(Specimen(specimen1.genotype_len, specimen1.genotype))
        else:
            new_specimens.append(Specimen(specimen2.genotype_len, specimen2.genotype))
    return new_specimens


def runMutations(mutation_probability, bit_change_probability, specimens:list[Specimen]):
    for specimen in specimens:
        if random() < mutation_probability:
            i = 0
            while i < len(specimen.genotype):
                if random() < bit_change_probability:
                    specimen.genotype[i] = not specimen.genotype[i]
                i += 1
    return specimens


if __name__ == "__main__":
    g = createRandomGraph(10, 0.5)
    pop = generatePopulation(5, 10)
    pop2 = rankSpecimens(g, pop)
    for specimen in pop2:
        print(f'{specimen.score}, {specimen.rank}, {specimen.genotype_as_string()}')
    drawGraph(g)
