import numpy as np
from utils.class_relationship_utils import find_parent

CLASS_SEPARATOR = '/'

def create_combinations(path, separator, combinations):

    for i in range(1,(len(path)+1)):
        combination = path[0:i]
        str_join = separator.join(combination)
        combinations.append(str_join)

    return combinations


def get_possible_classes(classes):
    combinations = []

    for i in range(len(classes)):
        possible_class = str(classes[i])
        class_splitted = possible_class.split(CLASS_SEPARATOR)
        combinations = create_combinations(class_splitted, CLASS_SEPARATOR, combinations)

    combinations = np.unique(combinations)
    print("Getting all possible combinations for each label: {}".format(combinations))

    return combinations