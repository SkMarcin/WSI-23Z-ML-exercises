from graphs import (createFullGraph,
    createRandomGraph,
    drawGraph,
    setGraphSeed,
    transformToNXGraph,
    countLitUnlit,
    choices
)


class Specimen:
    def __init__(self, genotype=[], genotype_len=50):
        self.genotype = genotype
        if self.genotype == []:
            self.generate_genotype(genotype_len)
        self.score = -1
        self.rank = -1
        self.reproduction_probability = 0

    def generate_genotype(self, genotype_len):
        for i in range(0, genotype_len):
            self.genotype.append(choices([True, False]))


def rankSpecimens(graph, specimens : list[Specimen]):
    for specimen in specimens:
        specimen.score = objectiveFunction(graph, specimen)
    specimens.sort(key=lambda x: x.score)
    i = 1
    for specimen in specimens:
        specimen.rank = i
        i += 1


def objectiveFunction(graph, specimen:Specimen):
    setGraphSeed(graph, specimen.genotype)
    G = transformToNXGraph(graph)
    lit, unlit = countLitUnlit(G)
    return lit - unlit


def runTournaments(specimens:list[Specimen]):
    n = 2 * len(list(specimens))
    new_specimens = []
    for _ in range (0, n):
        specimen1 = choices(specimens)
        specimens.remove(specimen1)
        specimen2 = choices(specimens)
        specimens.append(specimen1)
        if specimen1.rank < specimen2.rank:
            new_specimens.append(specimen1)
        else:
            new_specimens.append(specimen2)
    return new_specimens

def runMutations(mutation_probability, bit_change_probability, specimens:list[Specimen]):
    for specimen in specimens:
        if choices([True, False], weights=[mutation_probability, 1-mutation_probability]):
            for bit in specimen.genotype():
                if choices([True, False], weights=[bit_change_probability]):
                    bit = not bit
    return specimens

