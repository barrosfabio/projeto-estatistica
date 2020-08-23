from hierarchical_classifier.classification_pipeline.classification_pipeline import HierarchicalClassificationPipeline
from hierarchical_classifier.constants.resampling_constants import *
from hierarchical_classifier.results.global_results_framework import GlobalResultsFramework
from utils.directory_utils import create_result_directories
from utils.data_utils import load_csv_data


train_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_train.csv'
test_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_test.csv'
result_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/hierarchical_classifier/csv_results/experiment_result'
resampling_algorithm = 'smote'
classifier_name = 'rf'

results_list = []
resampling_algorithms = [SMOTE_RESAMPLE]
resampling_strategies = [FLAT_RESAMPLING]

create_result_directories(result_path)

# 1. Load the data from a CSV file
# 2. Get inputs and outputs
[train_data_frame, unique_train_classes] = load_csv_data(train_path)
[test_data_frame, unique_test_classes] = load_csv_data(test_path)

for strategy in resampling_strategies:

    if strategy != NONE:
        for algorithm in resampling_algorithms:

            classification_pipeline = HierarchicalClassificationPipeline(train_data_frame, test_data_frame, unique_train_classes, algorithm, classifier_name,
                                                                     strategy)

            result = classification_pipeline.run()
            results_list.append(result)
    else:
        classification_pipeline = HierarchicalClassificationPipeline(train_data_frame, test_data_frame, unique_train_classes, NONE, classifier_name,
                                                                     strategy)

        result = classification_pipeline.run()
        results_list.append(result)


result_framework = GlobalResultsFramework(result_path)
result_framework.transform_to_csv(results_list)