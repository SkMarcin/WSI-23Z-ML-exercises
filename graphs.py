import networkx as nx
import matplotlib.pyplot as plt
from numpy.random import choice


class Node:
    def __init__(self, number, is_used, neighbors=[]):
        self.number = number
        self.is_used = is_used
        self.neighbors = neighbors

    def mutate(self):
        self.is_used = not self.is_used


def createFullGraph(n):
    graph = []

    for i in range(0, n):
        nodes = [k for k in range(i + 1, n)]
        graph.append(Node(i, False, nodes))

    for i in range(0, n):
        for neighbor in i.neighbors:
            neighbor.neighbors.append(i)

    return graph


def createRandomGraph(n, edge_probability):
    graph = []

    for i in range(0, n):
        nodes = []
        for k in range(i + 1, n):
            if choice([True, False], p=[edge_probability, 1-edge_probability]):
                nodes.append(k)

        graph.append(Node(i, False, nodes))

    for i in range(0, n):
        for neighbor_num in graph[i].neighbors:
            graph[neighbor_num].neighbors.append(i)

    return graph


def setGraphSeed(graph, seed):
    for i in range(0, len(graph)):
        if seed[i]:
            graph[i].is_used = True
        else:
            graph[i].is_used = False


def transformToNXGraph(graph):
    G = nx.Graph()

    for i in range(0, len(graph)):
        node = graph[i]
        lit_edges = []
        unlit_edges = []

        if node.is_used:
            for neighbor in node.neighbors:
                lit_edges.append((i, neighbor))
        else:
            for neighbor in node.neighbors:
                if graph[neighbor].is_used:
                    lit_edges.append((i, neighbor))
                else:
                    unlit_edges.append((i, neighbor))

        G.add_edges_from(lit_edges, color='g')
        G.add_edges_from(unlit_edges, color='r')
    return G


def drawGraph(graph):
    G = transformToNXGraph(graph)
    colors_list = nx.get_edge_attributes(G, 'color').values()
    options = {
        "edge_color": colors_list,
        "width": 1,
        "with_labels": True,
    }
    nx.draw(G, **options)
    plt.show()


def countLitUnlit(G):
    lit = 0
    unlit = 0
    edges = G.edges.data("color")
    for edge in edges:
        if edge[2] == 'g':
            lit += 1
        else:
            unlit += 1
    return lit, unlit


#if __name__ == "main":
seed = []
for i in range(0, 50):
    seed.append(choice([0, 1]))
print(seed)
gg = createRandomGraph(50, 0.1)
setGraphSeed(gg, seed)
drawGraph(gg)


