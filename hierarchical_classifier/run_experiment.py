from hierarchical_classifier.classification_pipeline.classification_pipeline import HierarchicalClassificationPipeline
from hierarchical_classifier.constants.resampling_constants import *
from hierarchical_classifier.results.global_results_framework import GlobalResultsFramework

train_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_train.csv'
test_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_feature_matrix_test.csv'
result_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/hierarchical_classifier/experiment_result'
resampling_algorithm = 'smote'
classifier_name = 'rf'

results_list = []
resampling_algorithms = [SMOTE_RESAMPLE, SMOTE_ENN, SMOTE_TOMEK, ADASYN_RESAMPLER, RANDOM_OVERSAMPLER]
resampling_strategies = [NONE, FLAT_RESAMPLING]

for strategy in resampling_strategies:

    if strategy != NONE:
        for algorithm in resampling_algorithms:

            classification_pipeline = HierarchicalClassificationPipeline(train_path, test_path, algorithm, classifier_name,
                                                                     strategy)

            result = classification_pipeline.run()
            results_list.append(result)
    else:
        classification_pipeline = HierarchicalClassificationPipeline(train_path, test_path, NONE, classifier_name,
                                                                     strategy)

        result = classification_pipeline.run()
        results_list.append(result)


result_framework = GlobalResultsFramework(result_path)
result_framework.transform_to_csv(results_list)