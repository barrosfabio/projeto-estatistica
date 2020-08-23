from hierarchical_classifier.tree.generic_tree import Tree
from utils.data_utils import slice_data
from hierarchical_classifier.tree.data import Data
from hierarchical_classifier.classification.classification_algorithm import ClassificationAlgorithm
import numpy as np
from hierarchical_classifier.constants.resampling_constants import HIERARCHICAL_RESAMPLING
from hierarchical_classifier.resampling.resampling_algorithm import ResamplingAlgorithm


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

            # Resampling for each parent node data
            if self.resampling_strategy == HIERARCHICAL_RESAMPLING:
                unique_classes = np.unique(positive_classes_data.iloc[:,-1])
                if len(unique_classes) > 1:
                    resampling_algorithm = ResamplingAlgorithm(self.resampling_strategy, self.resampling_algorithm, 4, root_node.class_name)
                    positive_classes_data = resampling_algorithm.resample(positive_classes_data)

            # Slice data in inputs and outputs
            [input_data, output_data] = slice_data(positive_classes_data)

            # Relabel the outputs to the child classes
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

    def train_lcpn(self, root_node):

        # Testing if the node is leaf
        if len(root_node.child) == 0:
            print('This is a leaf node we do not need to train')
            return
        else:
            # Retrieve training data
            training_data = root_node.data

            # Train the classifier
            classifier = ClassificationAlgorithm(self.classification_algorithm)
            print('Started Training for parent node {}'.format(root_node.class_name))
            classifier.train(training_data)
            print('Finished Training')

            # Save the classifier in the tree
            print('Saving the classifer...')
            root_node.set_classifier(classifier)

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                print('Child is {}'.format(root_node.child[i].class_name))
                self.train_lcpn(root_node.child[i])

    def predict_lcpn(self, root_node, sample):

        # Testing if the node is leaf
        if len(root_node.child) == 0:
            print('Leaf node reached. Class: {}'.format(root_node.class_name))
            return root_node.class_name
        else:
            # Retrieve the classifier
            classifier = root_node.classifier
            print('Started Prediction...'.format(root_node.class_name))
            predicted_class = classifier.prediction(sample)

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to check which child was the prediction
            for i in range(children):
                child_class = root_node.child[i].class_name

                # When the correct child is found, we will continue calling the recursion
                if child_class == predicted_class:
                    print('Child predicted is {}'.format(child_class))
                    return self.predict_lcpn(root_node.child[i], sample)

    def predict_from_sample_lcpn(self, root_node, test_inputs):
        predicted_classes = []
        i = 0

        for sample in test_inputs:
            predicted_class = self.predict_lcpn(root_node, sample)

            predicted_classes.append(predicted_class)
            print('Record being predicted {}/{}'.format(i, len(test_inputs)))
            i += 1

        return predicted_classes
