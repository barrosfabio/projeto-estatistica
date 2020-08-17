class Node:
    is_parent = False
    is_leaf = False

    def __init__(self, class_name, data_class_relationship):
        self.class_name = class_name
        self.data_class_relationship = data_class_relationship
        self.child = []
        self.data = None

