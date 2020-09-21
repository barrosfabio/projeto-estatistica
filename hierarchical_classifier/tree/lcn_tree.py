from hierarchical_classifier.tree.generic_tree import find_node
from hierarchical_classifier.tree.generic_tree import Tree
from utils.data_utils import slice_data
from hierarchical_classifier.tree.data import Data
from hierarchical_classifier.classification.classification_algorithm import ClassificationAlgorithm
from hierarchical_classifier.constants.resampling_constants import HIERARCHICAL_RESAMPLING
from hierarchical_classifier.resampling.resampling_algorithm import ResamplingAlgorithm
from hierarchical_classifier.configurations.global_config import GlobalConfig
from hierarchical_classifier.constants.utils_constants import CLASS_SEPARATOR, NEGATIVE_CLASS, PREDICTION_CONFIG
import numpy as np
import math


def relabel_outputs_lcn(data_frame, data_class):
    unique_classes = np.unique(data_frame['class'])
    data_frame['class'] = data_frame['class'].replace(unique_classes, data_class)

    return data_frame

class LCNTree(Tree):

    def retrieve_data(self, root_node, data_frame):
        print('Currently retrieving data for class: {}'.format(root_node.class_name))

        # If the current node doesn't have child, it is a leaf node
        if len(root_node.child) == 0:
            print('Reached leaf node level, call is being returned.')
            return
        else:
            # Retrieve the positive classes for the current node
            data_class_relationship = root_node.data_class_relationship
            positive_classes = data_class_relationship.positive_classes
            negative_classes = data_class_relationship.negative_classes
            print('Positive classes {} for node {}'.format(positive_classes, root_node.class_name))
            print('Negative classes {} for node {}'.format(negative_classes, root_node.class_name))

            # Retrieve the filtered data from the data_frame
            positive_classes_data = data_frame[data_frame['class'].isin(positive_classes)]
            negative_classes_data = data_frame[data_frame['class'].isin(negative_classes)]

            # Relabel the positive classes
            positive_classes_data = relabel_outputs_lcn(positive_classes_data, root_node.class_name)

            # Relabel the negative classes
            negative_classes_data = relabel_outputs_lcn(negative_classes_data, NEGATIVE_CLASS)

            # Concatenate both positive and negative classes data-frames
            final_data_frame = np.concatenate((positive_classes_data, negative_classes_data))

            # Resample data
            if self.resampling_strategy == HIERARCHICAL_RESAMPLING:
                unique_classes = np.unique(final_data_frame.iloc[:, -1])
                if len(unique_classes) > 1:
                    global_config = GlobalConfig.instance()
                    k_neighbors = global_config.k_neighbors
                    resampling_algorithm = ResamplingAlgorithm(self.resampling_strategy, self.resampling_algorithm,
                                                               k_neighbors, root_node.class_name)
                    final_data_frame = resampling_algorithm.resample(final_data_frame, self.fold)

            # Slice data in inputs and outputs
            [input_data, output_data] = slice_data(final_data_frame)

            # Store the data in the node
            root_node.data = Data(input_data, output_data)

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                print('Child is {}'.format(root_node.child[i].class_name))
                self.retrieve_lcn_data(root_node.child[i], data_frame)

    def train_hierarchical(self, root_node):

        parent_node = root_node

        # Testing if the node is leaf
        if len(root_node.child) == 0:
            print('This is a leaf node we do not need to train')
            return
        else:

            # Retrieve child nodes
            children = len(root_node.child)

            # Train each child node
            for i in range(children):
                # Retrieve node being visited
                visited_node = root_node.child[i]
                print('Child is {}'.format(root_node.child[i].class_name))

                # Retrieve training data for the visited node
                training_data = visited_node.data

                # Train the classifier
                classifier = ClassificationAlgorithm(self.classification_algorithm)
                print('Started Training for parent node {}'.format(root_node.class_name))

                classifier.train(training_data)
                print('Finished Training')

                # Save the classifier in the tree
                print('Saving the classifer...')
                visited_node.set_classifier(classifier)

                # Go down the tree
                self.train_lcn(visited_node)

    def predict_hierarchical(self, root_node, sample):

        # Testing if the node is leaf
        if len(root_node.child) != 0:
            # Retrieve child
            children = len(root_node.child)

            # Arrays to store the predictions
            proba_dict = {}
            predicted_classes = []

            # Iterate over child
            for i in range(children):
                visited_node = root_node.child[i]
                print('Child is {}'.format(root_node.child[i].class_name))

                # Asking the trained classifier which class the data belongs
                classifier = visited_node.clf
                [predicted_class, probability] = classifier.prediction_proba(sample)

                # Saving the predicted class and probabilty
                predicted_classes = np.append(predicted_classes, predicted_class)
                proba_dict[visited_node.class_name] = probability

            # Add a method to count how many negative predictions we have
            negative_predictions = self.count_negative_predictions(predicted_classes)

            # If the number of negative predictions is equal to the size of children array - 1, that is the correct case
            if negative_predictions == (len(children) - 1):
                print('Normal Situation')
                predicted_class = self.find_predicted_class(predicted_classes)
                predicted_node = find_node(root_node, predicted_class)

                # Continue going down in the tree as we are looking for leaf nodes
                final_prediction = self.predict_lcn(predicted_node, sample)
            else:
                print('Untie with probability')
                untied_class = self.prediction_inconsistency_handler(proba_dict)

                # Test if we have non mandatory leaf node prediction. In that case we can return classes before it reaches leaf node level
                if (PREDICTION_CONFIG != 'mandatory-leaf'):
                    return untied_class
                else:
                    # Find which node of the tree was predicted
                    predicted_node = find_node(root_node, untied_class)

                    # Continue going down in the tree as we are looking for leaf nodes
                    final_prediction = self.predict_lcn(predicted_node, sample)

            return final_prediction

        else:
            print('Leaf node reached. Class: {}'.format(root_node.class_name))
            return root_node.class_name


    def predict_from_sample(self, root_node, test_inputs):
        predicted_classes = []
        i = 0

        for sample in test_inputs:
            predicted_class = self.predict_lcn(root_node, sample)

            predicted_classes.append(predicted_class)
            print('Record being predicted {}/{}'.format(i, len(test_inputs)))
            i += 1

        return predicted_classes

    def find_predicted_class(self, predictions):
        negative_class_string = NEGATIVE_CLASS
        predicted_class = ''

        for i in range(0, len(predictions)):
            if predictions[i] != negative_class_string:
                predicted_class = predictions[i]
        return predicted_class

    def count_negative_predictions(self, predictions):
        negative_class_string = NEGATIVE_CLASS
        count_negative = 0

        for i in range(0, len(predictions)):
            if predictions[i] == negative_class_string:
                count_negative += 1
        return count_negative

    def prediction_inconsistency_handler(self, probability_dictionary):
        predicted_class = ''
        max_probability = -math.inf

        for key, value in probability_dictionary.items():
            positive_class_prob = value[0][1]
            if positive_class_prob > max_probability:
                max_probability = positive_class_prob
                predicted_class = key
        return predicted_class