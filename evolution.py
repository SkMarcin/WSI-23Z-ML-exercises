from graphs import (createFullGraph,
    createRandomGraph,
    drawGraph,
    setGraphSeed,
    transformToNXGraph,
    countLitUnlit,
    choice
)


class Specimen:
    def __init__(self, genotype=[], genotype_len=50):
        self.genotype = genotype
        if self.genotype == []:
            self.generate_genotype(genotype_len)

    def generate_genotype(self, genotype_len):
        for i in range(0, genotype_len):
            self.genotype.append(choice([True, False]))



def objectiveFunction(G):
    lit, unlit = countLitUnlit(G)
    return lit - unlit

def selectTournaments(specimens:list):
    specimen1 = choice(specimens)
    specimens.remove(specimen1)
    specimen2 = choice(specimens)