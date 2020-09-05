from hierarchical_classifier.classification_pipeline.classification_pipeline import HierarchicalClassificationPipeline
from hierarchical_classifier.constants.resampling_constants import *
from hierarchical_classifier.results.global_results_framework import GlobalResultsFramework
from hierarchical_classifier.hierarchical_utils.hierarchical_results_utils import calculate_average_fold_result
from hierarchical_classifier.results.dto.average_experiment_result_dto import AverageExperimentResultDTO
from hierarchical_classifier.hierarchical_utils.directory_utils import create_result_directories
from utils.data_utils import slice_data, array_to_data_frame
from utils.data_utils import load_csv_data
from sklearn.model_selection import StratifiedKFold

path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_train.csv'
result_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/hierarchical_classifier/final_results/experiment_results'
classifier_name = 'rf'
folds = 5

results_list = []
results_per_class_list = []
resampling_algorithms = [SMOTE_RESAMPLE, SMOTE_ENN, SMOTE_TOMEK]
resampling_strategies = [NONE, FLAT_RESAMPLING]


def run_cross_validation(folds, input_data, output_data, unique_classes, algorithm, classifier_name, strategy):
    folds_result_list = []
    kfold = StratifiedKFold(n_splits=folds, shuffle=True)
    kfold_count = 1

    for train_index, test_index in kfold.split(input_data, output_data):
        print('----------Started fold {} ----------'.format(kfold_count))
        # Slice inputs and outputs
        input_data_train, output_data_train = input_data[train_index], output_data[train_index]
        input_data_test, output_data_test = input_data[test_index], output_data[test_index]

        # Transform fold_data into data_frame again
        train_data_frame = array_to_data_frame(input_data_train, output_data_train)
        test_data_frame = array_to_data_frame(input_data_test, output_data_test)

        # Build the classification experiment
        classification_pipeline = HierarchicalClassificationPipeline(train_data_frame, test_data_frame, unique_classes,
                                                                     algorithm, classifier_name,
                                                                     strategy, kfold_count)

        # Run and retrieve the result
        result = classification_pipeline.run()
        kfold_count += 1

        # Pipeline result being appended to the list of results for each fold
        folds_result_list.append(result)

    # Calculate the average result considering the k-folds
    [average_experiment_results, per_class_results_data_frame] = calculate_average_fold_result(folds_result_list, unique_classes)

    # Build DTO objects to return the average result
    average_result = AverageExperimentResultDTO(average_experiment_results, strategy, algorithm)
    average_result_per_class = AverageExperimentResultDTO(per_class_results_data_frame, strategy, algorithm)

    return [average_result, average_result_per_class]


# Creating the directories to store the results
create_result_directories(result_path, resampling_strategies, resampling_algorithms)

# Load the data from a CSV file
[data_frame, unique_classes] = load_csv_data(path)
[input_data, output_data] = slice_data(data_frame)

for strategy in resampling_strategies:

    if strategy != NONE:
        for algorithm in resampling_algorithms:
            [result, result_per_class] = run_cross_validation(folds, input_data, output_data, unique_classes, algorithm,
                                                              classifier_name, strategy)

            results_list.append(result)
            results_per_class_list.append(result_per_class)
    else:

        [result, result_per_class] = run_cross_validation(folds, input_data, output_data, unique_classes, NONE,
                                                          classifier_name, strategy)

        results_list.append(result)
        results_per_class_list.append(result_per_class)


result_framework = GlobalResultsFramework()
result_framework.transform_to_csv(results_list)
result_framework.transform_per_class_to_csv(results_per_class_list)






