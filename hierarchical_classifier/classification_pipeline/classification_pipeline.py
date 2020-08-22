from utils.data_utils import *
from hierarchical_classifier.tree.lcpn_tree import LCPNTree
from hierarchical_classifier.evaluation.hierarchical_metrics import calculate_hierarchical_metrics
from hierarchical_classifier.resampling.resampling_algorithm import ResamplingAlgorithm
from hierarchical_classifier.results.dto.final_result_dto import FinalResultDTO
from hierarchical_classifier.results.local_results_framework import LocalResultsFramework
from hierarchical_classifier.constants.resampling_constants import FLAT_RESAMPLING


class HierarchicalClassificationPipeline:

    def __init__(self, train_path, test_path, resampling_algorithm, classifier_name, resampling_strategy):
        self.train_path = train_path
        self.test_path = test_path
        self.resampling_algorithm = resampling_algorithm
        self.classifier_name = classifier_name
        self.resampling_strategy = resampling_strategy

    def run(self):
        # Steps to build a hierarchical classifier

        # 1. Load the data from a CSV file
        # 2. Get inputs and outputs
        [train_data_frame, unique_train_classes] = load_csv_data(self.train_path)
        [test_data_frame, unique_test_classes] = load_csv_data(self.test_path)

        # 3. From the outputs array, use it to build the class_tree and to get the positive and negative classes according to
        # a policy
        tree = LCPNTree(unique_train_classes, self.classifier_name, self.resampling_strategy, self.resampling_algorithm)
        class_tree = tree.build_tree()

        # 4. From the class_tree, retrieve the data for each node, based on the list of positive and negative classes
        # If FLAT_SAMPLING_STRATEGY is chosen, we will resample the training data here
        if self.resampling_strategy == FLAT_RESAMPLING:
            resampling_algorithm = ResamplingAlgorithm(self.resampling_strategy, self.resampling_algorithm, 4)
            train_data_frame = resampling_algorithm.resample(train_data_frame)

        tree.retrieve_lcpn_data(class_tree, train_data_frame)

        # 5. Train the classifiers
        tree.train_lcpn(class_tree)

        # 6. Predict
        [inputs_test, outputs_test] = slice_data(test_data_frame)
        predicted_classes = np.array(tree.predict_from_sample_lcpn(class_tree, inputs_test))

        # 7. Calculate the final/local results
        local_results = LocalResultsFramework(self.resampling_strategy, self.resampling_algorithm)
        local_results.calculate_perclass_metrics(outputs_test, predicted_classes)
        local_results.calculate_parent_node_metrics(class_tree, outputs_test, predicted_classes)
        local_results.save_to_csv()

        [hp, hr, hf] = calculate_hierarchical_metrics(predicted_classes, outputs_test)


        print('\n-------------------Results Summary-------------------')
        print('Hierarchical Precision: {}'.format(hp))
        print('Hierarchical Recall: {}'.format(hr))
        print('Hierarchical F-Measure: {}'.format(hf))
        print('Classification completed')

        return FinalResultDTO(hp, hr, hf, self.resampling_algorithm, self.resampling_strategy)
