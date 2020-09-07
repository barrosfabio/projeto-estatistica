from utils.data_utils import *
from hierarchical_classifier.tree.lcpn_tree import LCPNTree
from hierarchical_classifier.evaluation.hierarchical_metrics import calculate_hierarchical_metrics
from hierarchical_classifier.resampling.resampling_algorithm import ResamplingAlgorithm
from hierarchical_classifier.results.dto.experiment_result_dto import ExperimentResultDTO
from hierarchical_classifier.results.dto.result_dto import ResultDTO
from hierarchical_classifier.results.pipeline_results_framework import PipeleineResultsFramework
from hierarchical_classifier.constants.resampling_constants import FLAT_RESAMPLING
from sklearn.metrics import confusion_matrix


class HierarchicalClassificationPipeline:

    def __init__(self, unique_classes, resampling_algorithm, classifier_name, resampling_strategy, fold):
        self.unique_classes = unique_classes
        self.resampling_algorithm = resampling_algorithm
        self.classifier_name = classifier_name
        self.resampling_strategy = resampling_strategy

    def run(self, train_data_frame, test_data_frame, fold):

        # Steps to build a hierarchical classifier

        # 1. From the outputs array, use it to build the class_tree and to get the positive and negative classes according to
        # a policy
        tree = LCPNTree(self.unique_classes, self.classifier_name, self.resampling_strategy, self.resampling_algorithm)
        class_tree = tree.build_tree()

        # 2. From the class_tree, retrieve the data for each node, based on the list of positive and negative classes
        # If FLAT_SAMPLING_STRATEGY is chosen, we will resample the training data here
        if self.resampling_strategy == FLAT_RESAMPLING:
            resampling_algorithm = ResamplingAlgorithm(self.resampling_strategy, self.resampling_algorithm, 3)
            train_data_frame = resampling_algorithm.resample(train_data_frame, fold)

        tree.retrieve_lcpn_data(class_tree, train_data_frame)

        # 3. Train the classifiers
        tree.train_lcpn(class_tree)

        # 4. Predict
        [inputs_test, outputs_test] = slice_data(test_data_frame)
        predicted_classes = np.array(tree.predict_from_sample_lcpn(class_tree, inputs_test))

        # 5. Calculate the final/local results
        pipeline_results = PipeleineResultsFramework(self.resampling_strategy, self.resampling_algorithm)
        per_class_metrics = pipeline_results.calculate_perclass_metrics(outputs_test, predicted_classes)
        conf_matrix = confusion_matrix(outputs_test, predicted_classes)

        [hp, hr, hf] = calculate_hierarchical_metrics(predicted_classes, outputs_test)


        print('\n-------------------Results Summary-------------------')
        print('Hierarchical Precision: {}'.format(hp))
        print('Hierarchical Recall: {}'.format(hr))
        print('Hierarchical F-Measure: {}'.format(hf))
        print('Classification completed')

        return ExperimentResultDTO(ResultDTO(hp, hr, hf), per_class_metrics, conf_matrix, self.resampling_strategy, self.resampling_algorithm, fold)


