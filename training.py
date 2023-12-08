from decision_trees import DecisionTree, copy
from random import choice, randint
import numpy as np

N = 50
# array[row][column]
# array[[column, column, column], row, row]

def create_forest(data, targets, trees_amount):
    trees = []
    for i in range(trees_amount):
        chosen_data = []
        chosen_targets = []
        for j in range(N):
            index = randint(0, len(data) - 1)
            chosen_data.append(data[index])
            chosen_targets.append(targets[index])
    
        print(len(chosen_data[0]))
        attributes_selected = [False] * len(chosen_data[0])
        removed_atr = len(chosen_data[0]) - int(np.sqrt(len(chosen_data[0])))
        for atr in range(removed_atr):
            removed_index = randint(0, len(chosen_data[0]) - 1)
            attributes_selected[removed_index] = True
            
        tree = DecisionTree(chosen_data, chosen_targets)
        tree.build_tree(attributes_selected, None, None)
        trees.append(tree)

    return trees
    


