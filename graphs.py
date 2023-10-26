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


def create_full_graph(n):
    graph = []

    for i in range(0, n):
        nodes = [k for k in range(0, n)]
        nodes.remove(i)
        graph.append(Node(i, False, nodes))

    return graph

def create_random_graph(n, edge_probability):
    graph = []

    for i in range(0, n):
        nodes = []
        for k in range(0, n):
            if choice([True, False], p=[edge_probability, 1-edge_probability]):
                nodes.append(k)
        
        if i in nodes:
            nodes.remove(i)
        graph.append(Node(i, False, nodes))

    return graph

def draw_graph(graph):
    G = nx.Graph()

    for i in range(0, len(graph)):
        node = graph[i]
        edges = []
        for neighbor in node.neighbors:
            edges.append((i, neighbor))
        G.add_edges_from(edges)

    nx.draw_circular(G)
    plt.show()


if __name__ == "main":
    gg = create_random_graph(50, 0.1)
    draw_graph(gg)


