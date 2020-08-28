from hierarchical_classifier.classification_pipeline.classification_pipeline import HierarchicalClassificationPipeline
from hierarchical_classifier.constants.resampling_constants import *
from hierarchical_classifier.results.global_results_framework import GlobalResultsFramework
from utils.directory_utils import create_result_directories
from utils.data_utils import slice_data, array_to_data_frame
from utils.data_utils import load_csv_data
from sklearn.model_selection import KFold

path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_train.csv'
result_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/final_results/experiment_result'
classifier_name = 'rf'

results_list = []
resampling_algorithms = [SMOTE_RESAMPLE, SMOTE_ENN, SMOTE_TOMEK, RANDOM_OVERSAMPLER, RANDOM_UNDERSAMPLER, NEAR_MISS]
resampling_strategies = [NONE, FLAT_RESAMPLING, HIERARCHICAL_RESAMPLING]

create_result_directories(result_path, resampling_strategies)

# Load the data from a CSV file
[data_frame, unique_classes] = load_csv_data(path)
[input_data, output_data] = slice_data(data_frame)

for strategy in resampling_strategies:

    if strategy != NONE:
        for algorithm in resampling_algorithms:


                results_list.append(result)
    else:
        classification_pipeline = HierarchicalClassificationPipeline(train_data_frame, test_data_frame, unique_classes, NONE, classifier_name,
                                                                     strategy)

        result = classification_pipeline.run()
        results_list.append(result)


result_framework = GlobalResultsFramework()
result_framework.transform_to_csv(results_list)



def run_cross_validation_classifier(folds, train_data_frame, test_data_frame, unique_classes, algorithm, classifier_name, strategy):
    folds_result_list = []
    kfold = KFold(n_splits=5, shuffle=True)
    kfold_count = 1

    for train_index, test_index in kfold.split(input_data, output_data):
        print('----------Started fold {} ----------'.format(kfold_count))
        # Slice inputs and outputs
        input_data_train, output_data_train = input_data[train_index], output_data[train_index]
        input_data_test, output_data_test = input_data[test_index], output_data[test_index]

        # Transform fold_data into data_frame again
        train_data_frame = array_to_data_frame(input_data_train, output_data_train)
        test_data_frame = array_to_data_frame(input_data_train, output_data_train)

        classification_pipeline = HierarchicalClassificationPipeline(train_data_frame, test_data_frame, unique_classes,
                                                                     algorithm, classifier_name,
                                                                     strategy, kfold_count)

        result = classification_pipeline.run()
        kfold_count += 1

        # Pipeline result being appended to the list of results for each fold
        folds_result_list.append(result)

    # TODO: Calculate average result for each fold
