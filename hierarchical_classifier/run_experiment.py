from hierarchical_classifier.constants.resampling_constants import *
from hierarchical_classifier.results.global_results_framework import GlobalResultsFramework
from hierarchical_classifier.hierarchical_utils.directory_utils import create_result_directories
from hierarchical_classifier.constants.utils_constants import LCPN_CLASSIFIER, LCN_CLASSIFIER
from hierarchical_classifier.protocol.experimental_protocol import ExperimentalProtocol
from hierarchical_classifier.configurations.global_config import GlobalConfig
from utils.data_utils import slice_data
from utils.data_utils import load_csv_data

#path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/filtered_covid_canada_plus7.csv'
path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/filtered_covid_canada_rydles_plus7.csv'
#path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_train.csv'
result_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/hierarchical_classifier/final_results/experiment_results'
classifier_name = 'rf'
local_classifier_type = LCPN_CLASSIFIER
folds = 5
k_neighbors = 5

# Saving Global Configurations in singleton object
global_config = GlobalConfig.instance()
global_config.set_kneighbors(k_neighbors)
global_config.set_kfold(folds)
global_config.set_local_classifier(LCPN_CLASSIFIER)

results_list = []
results_per_class_list = []
resampling_algorithms = [RANDOM_OVERSAMPLER, SMOTE_RESAMPLE, BORDERLINE_SMOTE, ADASYN_RESAMPLER, SMOTE_ENN, SMOTE_TOMEK]
resampling_strategies = [NONE, FLAT_RESAMPLING]


# Creating the directories to store the results
create_result_directories(result_path, resampling_strategies, resampling_algorithms)

# Load the data from a CSV file
[data_frame, unique_classes] = load_csv_data(path)
[input_data, output_data] = slice_data(data_frame)

for strategy in resampling_strategies:

    if strategy != NONE:
        for algorithm in resampling_algorithms:
            experimental_protocol = ExperimentalProtocol(input_data, output_data, unique_classes, classifier_name)
            [result, result_per_class] = experimental_protocol.run_cross_validation(folds, algorithm, classifier_name, strategy)

            results_list.append(result)
            results_per_class_list.append(result_per_class)
    else:
        experimental_protocol = ExperimentalProtocol(input_data, output_data, unique_classes, classifier_name)
        [result, result_per_class] = experimental_protocol.run_cross_validation(folds, NONE, classifier_name,
                                                                                strategy)

        results_list.append(result)
        results_per_class_list.append(result_per_class)


result_framework = GlobalResultsFramework()
result_framework.transform_to_csv(results_list)
result_framework.transform_per_class_to_csv(results_per_class_list)






