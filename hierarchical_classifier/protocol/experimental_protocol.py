from hierarchical_classifier.classification_pipeline.classification_pipeline import HierarchicalClassificationPipeline
from hierarchical_classifier.hierarchical_utils.hierarchical_results_utils import calculate_average_fold_result
from hierarchical_classifier.results.dto.average_experiment_result_dto import AverageExperimentResultDTO
from utils.data_utils import slice_data, array_to_data_frame
from sklearn.model_selection import StratifiedKFold

class ExperimentalProtocol:

    def __init__(self, input_data, output_data, unique_classes, classifier_name):
        self.input_data = input_data
        self.output_data = output_data
        self.unique_classes = unique_classes
        self.classifier_name = classifier_name

    def run_cross_validation(self, folds, algorithm, classifier_name, strategy):

        folds_result_list = []
        kfold = StratifiedKFold(n_splits=folds, shuffle=True)
        kfold_count = 1


        for train_index, test_index in kfold.split(self.input_data, self.output_data):
            print('----------Started fold {} ----------'.format(kfold_count))
            # Slice inputs and outputs
            input_data_train, output_data_train = self.input_data[train_index], self.output_data[train_index]
            input_data_test, output_data_test = self.input_data[test_index], self.output_data[test_index]

            # Transform fold_data into data_frame again
            train_data_frame = array_to_data_frame(input_data_train, output_data_train)
            test_data_frame = array_to_data_frame(input_data_test, output_data_test)

            # Build the classification experiment
            classification_pipeline = HierarchicalClassificationPipeline(self.unique_classes,
                                                                         algorithm, classifier_name,
                                                                         strategy, kfold_count)

            # Run and retrieve the result
            result = classification_pipeline.run(train_data_frame, test_data_frame, kfold_count)
            kfold_count += 1

            # Pipeline result being appended to the list of results for each fold
            folds_result_list.append(result)

        # Calculate the average result considering the k-folds
        [average_experiment_results, per_class_results_data_frame] = calculate_average_fold_result(folds_result_list, self.unique_classes)

        # Build DTO objects to return the average result
        average_result = AverageExperimentResultDTO(average_experiment_results, strategy, algorithm)
        average_result_per_class = AverageExperimentResultDTO(per_class_results_data_frame, strategy, algorithm)

        return [average_result, average_result_per_class]