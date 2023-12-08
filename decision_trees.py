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
    def __init__(self, data, targets, max_depth):
        self.root = None
        self.max_depth = max_depth
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
                self.frequencies[target] = 0
                self.classes_dict[target] = 0
        
        self.total_entropy = 0
        for value in self.classes_dict:
            self.total_entropy -= self.classes_dict[value] / self.classes_sum * np.log10(self.classes_dict[value] / self.classes_sum)


    def fit(self, data, targets, depth):
        attributes_selected = False * self.atr_count
        self.root = self.build_tree(data, targets, attributes_selected, None, depth)


    def build_tree(self, data, targets, attributes_selected, parent_node, depth):
        for attribute in attributes_selected:
            if not attribute:
                all_selected = False
        
        if all_selected:
            car_attributes = None * self.atr_count
            node_checked = parent_node
            car_attributes[node_checked.atr_index] = node_checked.value
            while node_checked.parent is not None:
                node_checked = node_checked.parent
                car_attributes[node_checked.atr_index] = node_checked.value
            for row_index, row in enumerate(data):
                valid_columns = 0
                for column_index, column in enumerate(row):
                    if row[column_index] != car_attributes[column_index]:
                        break
                    valid_columns += 1
                if valid_columns == self.atr_count:
                    return Node(targets[row_index], None, None, attributes_selected, parent_node)

        if depth == 0:
            car_attributes = None * self.atr_count
            node_checked = parent_node
            car_attributes[node_checked.atr_index] = node_checked.value
            while node_checked.parent is not None:
                node_checked = node_checked.parent
                car_attributes[node_checked.atr_index] = node_checked.value

            for target in self.classes_dict:
                self.classes_dict[target] = 0
            for row_index, row in enumerate(data):
                valid_columns = 0
                for column_index, column in enumerate(row):
                    if row[column_index] != car_attributes[column_index] and car_attributes[column_index] is not None:
                        break
                    valid_columns += 1
                
                if valid_columns == self.atr_count:
                    self.classes_dict[targets[row_index]] += 1

            most_common_class = None
            max_amount = 0
            for target in self.classes_dict:
                if self.classes_dict[target >= max_amount]:
                    most_common_class = target
            return Node(most_common_class, None, None, attributes_selected, parent_node)

        inf_gains = []
        for atr_index, attribute in enumerate(attributes_selected):
            if attribute:
                inf_gains.append(None)
            else:
                inf_gains.append(self.total_entropy - self.get_attribute_entropy(data, targets, atr_index))
                

        max_index = -1
        max_value = float('-inf')
        for index, value in enumerate(inf_gains):
            if value is not None:
                if value > max_value:
                    max_value = value
                    max_index = index

        attributes_selected = copy.deepcopy(attributes_selected)
        attributes_selected[max_index] = True

        # max_index -> index of attribute to split with
        atr_values = {}
        for row_index, row in enumerate(data):
            if row[max_index] not in self.classes_dict:
                atr_values[row[max_index]] = 0
        
        node = Node(None, key, max_index, attributes_selected, parent_node)
        for key in atr_values:
            node.children.append(self.build_tree(data, targets, attributes_selected, node, depth - 1))
        return node
        

    def get_attribute_entropy(self, data, targets, atr_index):
        attribute_entropy = 0
        atr_sums = {}
        atr_frequencies = {}
        for row_index, row in enumerate(data):
            if row[atr_index] in self.classes_dict:
                atr_frequencies[(row[atr_index], targets[row_index])] += 1
                atr_sums[row[atr_index]] += 1
            else:
                atr_frequencies[(row[atr_index], targets[row_index])] = 0
                atr_sums[row[atr_index]] = 0


        for sum_key in atr_sums:
            temp_entropy = 0
            for freq_key in atr_frequencies:
                if freq_key[1] == sum_key:
                    temp_entropy -= atr_frequencies[freq_key] / atr_sums[sum_key] * np.log10(atr_frequencies[freq_key] / atr_sums[sum_key])
            attribute_entropy -= (temp_entropy * atr_sums[sum_key]) / self.classes_sum

        return attribute_entropy



