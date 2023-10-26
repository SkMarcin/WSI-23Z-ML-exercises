from random import choices

POPULATION_COUNT = 1000
MUTATION_PROBABILITY = 0.5
MAX_ITERATIONS = 500

print(choices([True, False], weights=[0.1, 0.9]))

def createRandomGraph(n, edge_probability):
    graph = []

    for i in range(0, n):
        nodes = []
        for k in range(i + 1, n):
            if choices([True, False], weights=[edge_probability, 1-edge_probability]):
                nodes.append(k)
        print(nodes)

    return graph

createRandomGraph(50, 0.1)