from copy import copy, deepcopy
from hierarchical_classifier.tree.node import Node
from hierarchical_classifier.policies.siblings_policy import SiblingsPolicy
from utils.tree_utils import get_possible_classes, find_parent


class Tree():

    def __init__(self, possible_classes):
        self.possible_classes = possible_classes

    def find_node(self, root, node_class):

        if root.class_name == node_class:
            return root
        else:
            children = len(root.child)

            for i in range(children):
                self.find_node(root.child[i], node_class)

    def insert_node(self, root, parent, data_class_relationship,  node_class):
        print('Node class: ' + node_class)
        print('Positive classes for node {}: {} '.format(node_class, data_class_relationship.positive_classes))
        print('Negative classes for node {}: {} '.format(node_class, data_class_relationship.negative_classes))
        print('Parent: ' + parent)

        if root is None:
            root = Node(node_class, data_class_relationship)
        else:
            print('Root class: ' + root.class_name)

            # If current root is the parent, then we add to its child
            if root.class_name == parent:
                root.child.append(Node(node_class, data_class_relationship))
            else:
                # If current root is not the parent, then we need to recursively go through the tree until we find its parent
                children = len(root.child)

                print("Opening each branch of the tree (each child)")

                for i in range(children):
                    child_updated = self.insert_node(root.child[i], parent,  data_class_relationship, node_class)
                    root.child[i] = child_updated

        return root

    def build_tree(self):
        root = None
        combinations = get_possible_classes(self.possible_classes)
        print("Number of possible classes: {}".format(len(combinations)))
        print("Starting to build a tree with all the possible classes...")

        for i in range(len(combinations)):
            # Insert current node
            current_node = combinations[i]

            # Identify parent
            parent = find_parent(current_node)

            # Builds an object to store the positive, negative classes and the direct_child of a given class
            data_classes_relationship = SiblingsPolicy(current_node, parent)

            # Identify Immediate child of the current class
            data_classes_relationship.find_direct_child(combinations)

            # Identify Positive and Negative Classes
            data_classes_relationship.find_classes_siblings_policy(combinations)

            # Insert the node in the tree
            root = self.insert_node(root, parent, data_classes_relationship, current_node)

        return root
