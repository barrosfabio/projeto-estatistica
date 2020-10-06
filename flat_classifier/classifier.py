import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flat_classifier import some_functions, resampling_algorithm
from hierarchical_classifier.evaluation import hierarchical_metrics
# Options
from flat_classifier.class_for_array import AllResamples
from sklearn.metrics import confusion_matrix
from utils.results_utils import plot_confusion_matrix

from datetime import datetime

from hierarchical_classifier.configurations.global_config import GlobalConfig
from hierarchical_classifier.hierarchical_utils.directory_utils import create_result_directories
from hierarchical_classifier.constants.resampling_constants import *

from hierarchical_classifier.results.pipeline_results_framework import PipeleineResultsFramework
from hierarchical_classifier.evaluation.flat_metrics import calculate_flat_metrics
from hierarchical_classifier.results.dto.experiment_result_dto import ExperimentResultDTO
from hierarchical_classifier.results.dto.result_dto import ResultDTO
from hierarchical_classifier.hierarchical_utils.hierarchical_results_utils import calculate_average_fold_result
from hierarchical_classifier.results.dto.average_experiment_result_dto import AverageExperimentResultDTO
from hierarchical_classifier.results.global_results_framework import GlobalResultsFramework


def main():
    data = '../feature_extraction/result/filtered_covid_canada_rydles_plus7.csv'
    classifier_name = 'knn'
    n_experiment_runs = 20
    n_folds = 5
    k_neighbors = 5
    result_path = 'results/detailed'

    global_config = GlobalConfig.instance()
    global_config.set_kneighbors(k_neighbors)
    global_config.set_kfold(n_folds)

    data_frame = pd.read_csv(data)

    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)

    [input_data, output_data] = some_functions.slice_data(data_frame)
    classes = np.unique(output_data)

    run_all_experiments(n_experiment_runs, result_path, output_data, input_data, kfold, classes, classifier_name)


def train_and_predict(train_index, test_index, input_data, output_data, resampling_algorithm_name, classifier_name):
    inputs_train, outputs_train = input_data[train_index], output_data[train_index]
    inputs_test, outputs_test = input_data[test_index], output_data[test_index]

    input_data_train = None
    output_data_train = None
    if resampling_algorithm_name != resampling_algorithm.NONE:
        print(f"Resampling data with {resampling_algorithm_name}")
        resampler = resampling_algorithm.instantiate_resampler('flat', resampling_algorithm_name, 5)

        input_data_train, output_data_train = resampler.fit_resample(inputs_train,
                                                                     outputs_train)
        print("Done resampling")
    else:
        input_data_train = inputs_train
        output_data_train = outputs_train

    print("Started training ", end='')

    clf = some_functions.get_classifier(classifier_name)
    clf = clf.fit(input_data_train, output_data_train)
    print("-> Finished")

    print("Start Prediction ", end='')
    predicted = clf.predict(inputs_test)
    print("-> Finished")

    return outputs_test, predicted


def run_cross_validation(output_data, input_data, kfold, classes, resampling_strategy, resampling_algorithm_name,
                         classifier_name):
    all_folds_result_list = []
    all_folds_outputs_test = []
    all_folds_predicted = []

    kfold_count = 1
    for train_index, test_index in kfold.split(input_data, output_data):
        outputs_test, predicted = train_and_predict(train_index, test_index, input_data, output_data,
                                                    resampling_algorithm_name, classifier_name)

        pipeline_results = PipeleineResultsFramework(resampling_strategy, resampling_algorithm_name)
        per_class_metrics = pipeline_results.calculate_perclass_metrics(outputs_test, predicted)
        conf_matrix = confusion_matrix(outputs_test, predicted)

        [hp, hr, hf] = calculate_flat_metrics(predicted, outputs_test, avg_type='macro')

        result = ExperimentResultDTO(ResultDTO(hp, hr, hf), per_class_metrics, conf_matrix, resampling_strategy,
                                     resampling_algorithm_name, kfold_count)

        print('=' * 50)
        print('')

        kfold_count += 1
        all_folds_result_list.append(result)

    [average_experiment_results, per_class_results_data_frame] = calculate_average_fold_result(all_folds_result_list,
                                                                                               classes)

    average_result = AverageExperimentResultDTO(average_experiment_results, resampling_strategy,
                                                resampling_algorithm_name)
    average_result_per_class = AverageExperimentResultDTO(per_class_results_data_frame, resampling_strategy,
                                                          resampling_algorithm_name)

    return [average_result, average_result_per_class]


def run_all_experiments(n_experiment_runs, result_path, output_data, input_data, kfold, classes, classifier_name):
    for experiment_index in range(1, n_experiment_runs + 1):

        results_list = []
        results_per_class_list = []
        create_result_directories(result_path + "_exp" + str(experiment_index),
                                  resampling_algorithm.resampling_strategies,
                                  resampling_algorithm.resampling_algorithms)

        for res, resampling_algorithm_name in enumerate(resampling_algorithm.resampling_algorithms):
            if resampling_algorithm_name == resampling_algorithm.NONE:
                resampling_strategy = resampling_algorithm.NONE
            else:
                resampling_strategy = resampling_algorithm.FLAT_RESAMPLING
            [average_result, average_result_per_class] = run_cross_validation(output_data, input_data, kfold, classes,
                                                                              resampling_strategy,
                                                                              resampling_algorithm_name,
                                                                              classifier_name)

            results_list.append(average_result)
            results_per_class_list.append(average_result_per_class)

        result_framework = GlobalResultsFramework()
        result_framework.transform_to_csv(results_list)
        result_framework.transform_per_class_to_csv(results_per_class_list)


main()
