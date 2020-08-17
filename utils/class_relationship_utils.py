import numpy as np

CLASS_SEPARATOR = '/'

def find_parent(possible_class):
    if len(possible_class) > 1:

        possible_class_splitted = possible_class.split(CLASS_SEPARATOR)
        possible_class_splitted = possible_class_splitted[0:-1]
        parent = CLASS_SEPARATOR.join(possible_class_splitted)

    # Parent is the root of the tree
    else:
        parent = 'R'
    return parent


def compare_child_length(possible_class, comparison):
    possible_class_splitted = possible_class.split(CLASS_SEPARATOR)
    comparison_list = comparison.split(CLASS_SEPARATOR)

    if len(comparison_list) == (len(possible_class_splitted) + 1):
        return True
    else:
        return False


def find_immediate_child(possible_class, combinations):
    combinations = list(combinations)
    immediate_child = {value for value in combinations if value.find(possible_class + CLASS_SEPARATOR) != -1 and compare_child_length(possible_class, value)}

    return list(immediate_child)


def find_sibling_classes(combinations, data_class, parent):
    # Classes at the same level
    same_level_classes = {value for value in combinations if
                          len(value.split(CLASS_SEPARATOR)) == len(data_class.split(CLASS_SEPARATOR))}

    # Classes at the same level and that share the same parent
    siblings = {value for value in same_level_classes if
                find_parent(value) == parent}

    return list(siblings)


def find_sibling_child(combinations, sibling):
    # Tries to find child classes for each sibling. Tests if the value is equal to sibling class + /
    child_classes = {value for value in combinations if value.find(sibling + CLASS_SEPARATOR) != -1}

    return list(child_classes)
