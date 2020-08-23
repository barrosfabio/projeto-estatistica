from hierarchical_classifier.results.results_framework import ResultsFramework
from hierarchical_classifier.results.dto.local_result_dto import LocalResultDTO
from hierarchical_classifier.evaluation.hierarchical_metrics import calculate_hierarchical_metrics
from hierarchical_classifier.configurations.global_config import GlobalConfig
from utils.results_utils import write_csv
import numpy as np
import pandas as pd
import os


class LocalResultsFramework(ResultsFramework):

    def __init__(self, resample_strategy, resample_algorithm):
        global_config = GlobalConfig.instance()
        local_result_path = global_config.directory_list['per_pipeline']
        self.resample_strategy = resample_strategy
        self.resample_algorithm = resample_algorithm
        self.per_class_results = []
        self.per_parent_metrics = []
        super(LocalResultsFramework, self).__init__(local_result_path)

    def filter_output_arrays(self, output_array, predicted_array, positive_classes):
        print('Positive classes: {}'.format(positive_classes))
        idx_filtered = []
        filtered_output_array = []
        filtered_predicted_array = []

        for current_class in positive_classes:


            # Filter the samples for the expected class
            expected_filtered = np.where(output_array == current_class)

            # Get the indexes of the filtered samples
            idx_filtered += (expected_filtered[0].tolist())

            # Samples that belong to the current_class filtered
            filtered_output_array = output_array[idx_filtered]

            # Filtering the same indexes in the predicted_array
            filtered_predicted_array = predicted_array[idx_filtered]

        return [filtered_output_array, filtered_predicted_array]

    def calculate_local_metrics(self, filtered_predicted_array, filtered_output_array):
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

        # Do it for each expected class
        for current_class in unique_classes:
            positive_classes = [current_class]

            [filtered_output_array, filtered_predicted_array] = self.filter_output_arrays(output_array, predicted_array, positive_classes)

            # Calculate metrics for current class
            print('Results Summary for class {}'.format(current_class))
            [hp, hr, hf] = self.calculate_local_metrics(filtered_predicted_array, filtered_output_array)

            self.per_class_results.append(LocalResultDTO(hp, hr, hf, current_class))

        #TODO: Add call to print to CSV here


    def calculate_parent_node_metrics(self, root_node, output_array, predicted_array):

        # Testing if the node is leaf
        if len(root_node.child) == 0:
            print('This is a leaf node we do not need to calculate anything')
            return
        else:
            # Retrieve positive classes for that node
            positive_classes = root_node.data_class_relationship.positive_classes
            current_class = root_node.class_name

            # Filtering the output array according to the positive classes
            [filtered_output_array, filtered_predicted_array] = self.filter_output_arrays(output_array,
                                                                                          predicted_array,
                                                                                          positive_classes)
            # Calculate metrics for parent node
            print('Results summary for parent class: {}'.format(current_class))
            [hp, hr, hf] = self.calculate_local_metrics(filtered_predicted_array, filtered_output_array)

            # Save per parent result
            self.per_parent_metrics.append(LocalResultDTO(hp, hr, hf, current_class))

            # Retrieve the number of children for the current node
            children = len(root_node.child)
            print('Current Node {} has {} child/children'.format(root_node.class_name, children))

            # Iterate over the current node child to call recursively for all of them
            for i in range(children):
                print('Child is {}'.format(root_node.child[i].class_name))
                self.calculate_parent_node_metrics(root_node.child[i], output_array, predicted_array)

    def list_to_data_frame(self, result_list):
        data_frame = pd.DataFrame()

        for result in result_list:
            row = {'hp': result.hp, 'hr': result.hp, 'hf': result.hf, 'class_name': result.class_name}
            data_frame = data_frame.append(row, ignore_index=True)

        return data_frame

    def save_to_csv(self):
        per_class_data_frame = self.list_to_data_frame(self.per_class_results)
        per_parent_data_frame = self.list_to_data_frame(self.per_parent_metrics)

        global_config = GlobalConfig.instance()
        per_class_path = global_config.directory_list['per_class_' + self.resample_strategy]
        per_parent_path = global_config.directory_list['per_parent_node_' + self.resample_strategy]

        write_csv(per_class_path + '/per_class_metrics_' + self.resample_algorithm, per_class_data_frame)
        write_csv(per_parent_path + '/per_parent_metrics_' + self.resample_algorithm, per_parent_data_frame)

