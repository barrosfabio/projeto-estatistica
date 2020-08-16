import numpy as np
from utils.class_relationship_utils import identify_parent

def create_combinations(path, separator, combinations):

    for i in range(1,(len(path)+1)):
        combination = path[0:i]
        str_join = separator.join(combination)
        combinations.append(str_join)

    return combinations


def get_possible_classes(classes):
    combinations = []

    if '.' in classes[1]:
        separator = '.'
    elif '/' in classes[1]:
        separator = '/'

    for i in range(len(classes)):
        possible_class = str(classes[i])
        class_splitted = possible_class.split(separator)
        combinations = create_combinations(class_splitted, separator, combinations)

    combinations = np.unique(combinations)
    print("Getting all possible combinations for each label: {}".format(combinations))

    return combinations


def build_tree(classes, tree):
    root = None
    combinations = get_possible_classes(classes)
    print("Number of possible classes: {}".format(len(combinations)))
    print("Starting to build a tree with all the possible classes...")

    for i in range(len(combinations)):
        # Insert current node
        current_node = combinations[i]

        # Identify parent
        parent = identify_parent(current_node)

        root = tree.insert_node(parent, current_node)


    return tree