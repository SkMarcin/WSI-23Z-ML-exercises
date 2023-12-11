from decision_trees import DecisionTree, copy
from random import choice, randint
import numpy as np

N = 250
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
    
        attributes_selected = [False] * len(chosen_data[0])
        removed_atr = len(chosen_data[0]) - int(np.sqrt(len(chosen_data[0])))
        for atr in range(removed_atr):
            removed_index = randint(0, len(chosen_data[0]) - 1)
            attributes_selected[removed_index] = True
            
        tree = DecisionTree(chosen_data, chosen_targets)
        tree.build_tree(attributes_selected, None, None)
        trees.append(tree)

    return trees
    
def get_forest_classifier(forest, attributes):
    classes_dict = {}
    for tree in forest:
        classifier = tree.get_prediction(attributes)
        if classifier is None:
            continue
        if classifier in classes_dict:
            classes_dict[classifier] += 1
        else:
            classes_dict[classifier] = 1
    
    max_classifier = None
    max_value = float('-inf')
    for classifier in classes_dict:
        if classes_dict[classifier] > max_value:
            max_classifier = classifier
            max_value = classes_dict[classifier]
    return max_classifier


def get_classification_dict(tuples):
    dict = {('unacc', 'unacc'): 0, ('unacc', 'acc'): 0, ('unacc', 'good'): 0, ('unacc', 'vgood'): 0, 
            ('acc', 'unacc'): 0, ('acc', 'acc'): 0, ('acc', 'good'): 0, ('acc', 'vgood'): 0,
            ('good', 'unacc'): 0, ('good', 'acc'): 0, ('good', 'good'): 0, ('good', 'vgood'): 0,
            ('vgood', 'unacc'): 0, ('vgood', 'acc'): 0, ('vgood', 'good'): 0, ('vgood', 'vgood'): 0}
    for tuple in tuples:
        dict[tuple] += 1
    return dict

def get_accuracy(dictionary, classes):
    correct = 0
    incorrect = 0
    for classifier in classes:
        for key in dictionary:
            if key[0] == key[1]:
                correct += dictionary[key]
            else:
                incorrect += dictionary[key]
    return correct / (correct + incorrect)

def get_recall(dictionary, classes):
    recalls = []
    for classifier in classes:
        correct = 0
        total = 0
        for key in dictionary:
            if key[0] == classifier:
                if key[1] == classifier:
                    correct += dictionary[key]
                total += dictionary[key]
        if total > 0:
            recalls.append(correct/total)
    return sum(recalls) / len(recalls)

def get_precision(dictionary, classes):
    precisions = []
    for classifier in classes:
        correct = 0
        total = 0
        for key in dictionary:
            if key[1] == classifier:
                if key[0] == classifier:
                    correct += dictionary[key]
                total += dictionary[key]
        if total > 0:
            precisions.append(correct/total)
    return sum(precisions) / len(precisions)