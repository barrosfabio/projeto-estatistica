
def split_compare(possible_class, comparison, separator):

    possible_class_splitted = possible_class.split(separator)
    comparison_list = comparison.split(separator)

    if len(comparison_list) == (len(possible_class_splitted)+1):
        return True
    else:
        return False


def identify_immediate_child(possible_class, combinations):
    child = []
    combinations = list(combinations)

    if '.' in combinations[1]:
        separator = '.'
    elif '/' in combinations[1]:
        separator = '/'

    for i in range(len(combinations)):
        comparison = str(combinations[i])
        if(comparison.find(possible_class + separator) != -1 and split_compare(possible_class, comparison, separator)):
            child.append(combinations[i])

    return child


def identify_parent(possible_class):
    if(len(possible_class)>1):
        if '.' in possible_class:
            separator = '.'
        elif '/' in possible_class:
            separator = '/'

        possible_class_splitted = possible_class.split(separator)

        possible_class_splitted = possible_class_splitted[0:-1]

        parent = separator.join(possible_class_splitted)
    # Parent is the root of the tree
    else:
        parent = 'R'
    return parent
