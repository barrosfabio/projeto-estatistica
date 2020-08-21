from hierarchical_classifier.results.results_framework import ResultsFramework
from hierarchical_classifier.results.result_dto import ResultDTO
from hierarchical_classifier.evaluation.hierarchical_metrics import calculate_hierarchical_metrics
import numpy as np

class LocalResultsFramework(ResultsFramework):

    def filter_output_arrays(self, output_array, predicted_array, positive_classes):
        print('Positive classes: {}'.format(positive_classes))

        for current_class in positive_classes:


            # Filter the samples for the expected class
            expected_filtered = np.where(output_array == current_class)

            # Get the indexes of the filtered samples
            idx_filtered = (expected_filtered[0].tolist())

            # Samples that belong to the current_class filtered
            filtered_output_array = output_array[idx_filtered]

            # Filtering the same indexes in the predicted_array
            filtered_predicted_array = predicted_array[idx_filtered]

        return [filtered_output_array, filtered_predicted_array]

    def calculate_local_metrics(self, current_class, filtered_predicted_array, filtered_output_array):
        if len(filtered_output_array) != 0:
            # Calculating the metrics for the current_class
            [hp, hr, hf] = calculate_hierarchical_metrics(filtered_predicted_array, filtered_output_array)
        else:
            hp = 0.0
            hr = 0.0
            hf = 0.0

        print('Hierarchical Precision:  {}'.format(hp))
        print('Hierarchical Recall:  {}'.format(hr))
        print('Hierarchical F-Measure:  {}'.format(hf))

        return [hp, hr, hf]

    def calculate_perclass_metrics(self, output_array, predicted_array):
        unique_classes = np.unique(output_array)
        per_class_metrics = {}

        # Do it for each expected class
        for current_class in unique_classes:
            positive_classes = [current_class]

            [filtered_output_array, filtered_predicted_array] = self.filter_output_arrays(self, output_array, predicted_array, positive_classes)

            # Calculate metrics for current class
            print('Results Summary for class {}'.format(current_class))
            [hp, hr, hf] = self.calculate_local_metrics(current_class, filtered_predicted_array, filtered_output_array)

            per_class_metrics[current_class].append(ResultDTO(hp, hr, hf))

    def calculate_parent_node_metrics(self, root_node, output_array, predicted_array, per_parent_metrics):

        # Testing if the node is leaf
        if len(root_node.child) == 0:
            print('This is a leaf node we do not need to calculate anything')
            return
        else:
            # Retrieve positive classes for that node
            positive_classes = root_node.positive_classes
            current_class = root_node.class_name

            # Filtering the output array according to the positive classes
            [filtered_output_array, filtered_predicted_array] = self.filter_output_arrays(self, output_array,
                                                                                          predicted_array,
                                                                                          positive_classes)
            # Calculate metrics for parent node
            print('Results summary for parent class: {}'.format(current_class))
            [hp, hr, hf] = self.calculate_local_metrics(current_class, filtered_predicted_array, filtered_output_array)

            # Save per parent result
            per_parent_metrics.append(ResultDTO(hp, hr, hf))

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                print('Child is {}'.format(root_node.child[i].class_name))
                self.calculate_parent_node_metrics(root_node.child[i], output_array, predicted_array, per_parent_metrics)