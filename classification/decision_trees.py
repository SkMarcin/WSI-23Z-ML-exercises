from random import choice
import numpy as np
import copy


class Node:
    def __init__(self, classifier, value, atr_index, attributes_selected, parent):
        self.classifier = classifier
        self.value = value
        self.atr_index = atr_index
        self.attributes_selected = attributes_selected
        self.parent = parent
        self.children = []

class DecisionTree:
    def __init__(self, data, targets):
        self.root = None
        self.data = data
        self.atr_count = len(data[0])
        self.targets = targets
        self.classes_dict = {}
        self.frequencies = {}
        self.classes_sum = 0

        for target in targets:
            if target in self.classes_dict:
                self.classes_sum += 1
                self.frequencies[target] += 1
            else:
                self.frequencies[target] = 1
                self.classes_dict[target] = 1

        self.total_entropy = 0
        for value in self.classes_dict:
            if self.frequencies[value] > 0:
                self.total_entropy -= (self.frequencies[value] / self.classes_sum) * (np.log10(self.frequencies[value] / self.classes_sum))


    def build_tree(self, attributes_selected, parent_node, split_value):
        all_selected = True
        for attribute in attributes_selected:
            if not attribute:
                all_selected = False

        car_attributes = [None] * self.atr_count
        node_checked = parent_node
        if node_checked is not None:
            car_attributes[node_checked.atr_index] = split_value
            while node_checked.parent is not None:
                car_attributes[node_checked.parent.atr_index] = node_checked.value
                node_checked = node_checked.parent


        for target in self.classes_dict:
            self.classes_dict[target] = 0
        for row_index, row in enumerate(self.data):
            valid_columns = 0
            for column_index, column in enumerate(row):
                if row[column_index] != car_attributes[column_index] and car_attributes[column_index] is not None:
                    break
                valid_columns += 1

            if valid_columns == self.atr_count:
                self.classes_dict[self.targets[row_index]] += 1

        class_counter = 0
        classifier = None
        for key in self.classes_dict:
            if self.classes_dict[key] > 0:
                class_counter += 1
                classifier = key

        if class_counter == 1:
            return Node(classifier, split_value, None, attributes_selected, parent_node)

        elif all_selected:
            most_common_class = None
            max_amount = 0
            for target in self.classes_dict:
                if self.classes_dict[target] >= max_amount:
                    most_common_class = target
            return Node(most_common_class, split_value, None, attributes_selected, parent_node)

        inf_gains = []
        for atr_index, attribute in enumerate(attributes_selected):
            if attribute:
                inf_gains.append(None)
            else:
                inf_gains.append(self.total_entropy - self.get_attribute_entropy(atr_index))

        max_index = -1
        max_value = float('-inf')
        for index, value in enumerate(inf_gains):
            if value is not None:
                if value > max_value:
                    max_value = value
                    max_index = index

        new_attributes_selected = copy.deepcopy(attributes_selected)
        new_attributes_selected[max_index] = True

        # max_index -> index of attribute to split with
        atr_values = {}
        for row_index, row in enumerate(self.data):
            if row[max_index] not in self.classes_dict:
                atr_values[row[max_index]] = 0

        node = Node(None, split_value, max_index, new_attributes_selected, parent_node)
        for split_value in atr_values:
            node.children.append(self.build_tree(new_attributes_selected, node, split_value))

        if parent_node == None:
            self.root = node

        return node


    def get_attribute_entropy(self, atr_index):
        attribute_entropy = 0
        atr_sums = {}
        atr_frequencies = {}
        for row_index, row in enumerate(self.data):
            if row[atr_index] in atr_sums:
                atr_sums[row[atr_index]] += 1
            else:
                atr_sums[row[atr_index]] = 1

            if (row[atr_index], self.targets[row_index]) in atr_frequencies:
                atr_frequencies[(row[atr_index], self.targets[row_index])] += 1
            else:
                atr_frequencies[(row[atr_index], self.targets[row_index])] = 1

        for sum_key in atr_sums:
            temp_entropy = 0
            for freq_key in atr_frequencies:
                if freq_key[0] == sum_key:
                    temp_entropy -= (atr_frequencies[freq_key] / atr_sums[sum_key]) * (np.log10(atr_frequencies[freq_key] / atr_sums[sum_key]))
            attribute_entropy -= (temp_entropy * atr_sums[sum_key]) / self.classes_sum

        return attribute_entropy


    def get_prediction(self, attributes):
        node = self.root
        if node is None:
            return None
        while node.atr_index is not None:
            found_child = False
            for child_node in node.children:
                if child_node.value == attributes[node.atr_index]:
                    node = child_node
                    found_child = True
                    break
            if not found_child:
                return None
        return node.classifier


