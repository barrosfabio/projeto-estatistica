from hierarchical_classifier.tree.generic_tree import Tree
from utils.data_utils import slice_data
from hierarchical_classifier.tree.data import Data
import numpy as np

CLASS_SEPARATOR = '/'


def relabel_outputs_lcpn(output_data, data_class):
    class_splitted = data_class.split(CLASS_SEPARATOR)
    relabeled_outputs = []

    for sample in output_data:
        sample_splitted = sample.split(CLASS_SEPARATOR)
        sample_splitted = sample_splitted[0:len(class_splitted) + 1]

        relabeled_sample = CLASS_SEPARATOR.join(sample_splitted)
        relabeled_outputs.append(relabeled_sample)

    return np.asarray(relabeled_outputs)


class LCPNTree(Tree):

    def retrieve_lcpn_data(self, root_node, data_frame):
        print('Currently retrieving data for class: {}'.format(root_node.class_name))

        # If the current node doesn't have child, it is a leaf node
        if len(root_node.child) == 0:
            print('Reached leaf node level, call is being returned.')
            return
        else:
            # Retrieve the positive classes for the current node
            data_class_relationship = root_node.data_class_relationship
            positive_classes = data_class_relationship.positive_classes
            print('Positive classes {} for node {}'.format(positive_classes, root_node.class_name))

            # Retrieve the filtered data from the data_frame
            positive_classes_data = data_frame[data_frame['class'].isin(positive_classes)]

            # Slice data in inputs and outputs
            [input_data, output_data] = slice_data(positive_classes_data)

            # Relabel the outputs
            output_data = relabel_outputs_lcpn(output_data, root_node.class_name)

            # Store the data in the node
            root_node.data = Data(input_data, output_data)

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                print('Child is {}'.format(root_node.child[i].class_name))
                self.retrieve_lcpn_data(root_node.child[i], data_frame)

