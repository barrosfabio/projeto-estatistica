import numpy as np
import pandas as pd
from hierarchical_classifier.results.dto.result_dto import ResultDTO
from hierarchical_classifier.configurations.global_config import GlobalConfig
from utils.results_utils import plot_confusion_matrix

def parse_result_metrics_list(fold_result_list):
    result_metrics_list = []

    for result in fold_result_list:
        result_metrics_list.append(result.result_metrics)
    return result_metrics_list

def parse_result_metrics_per_class_list(fold_result_list):
    per_class_metrics_list = []
    for result in fold_result_list:
        per_class_metrics_list.append(result.result_metrics)

    return per_class_metrics_list

def parse_confusion_matrix_list(fold_result_list):
    confusion_matrix_list = []

    for result in fold_result_list:
        confusion_matrix_list.append(result.conf_matrix)

    return confusion_matrix_list

def calculate_average_metrics(result_metrics_list):
    hp = []
    hr = []
    hf = []

    for result_metrics in result_metrics_list:
        hp.append(result_metrics.hp)
        hr.append(result_metrics.hr)
        hf.append(result_metrics.hf)

    return [np.mean(hp), np.mean(hr), np.mean(hf)]

def calculate_average_metrics_per_class(list_results, unique_classes):

    hp_dictionary = {}
    hr_dictionary = {}
    hf_dictionary = {}

    # Initialize dictionaries
    for unique in unique_classes:
        hp_dictionary[unique] = []
        hr_dictionary[unique] = []
        hf_dictionary[unique] = []

    # Write results
    for fold_result in list_results:
        for obj in fold_result:
            hp_dictionary[obj.class_name].append(obj.hp)
            hr_dictionary[obj.class_name].append(obj.hr)
            hf_dictionary[obj.class_name].append(obj.hf)

    # Calculating avg for each class
    for unique in unique_classes:
        hp_dictionary[unique] = np.mean(hp_dictionary[unique])
        hr_dictionary[unique] = np.mean(hr_dictionary[unique])
        hf_dictionary[unique] = np.mean(hf_dictionary[unique])

    # Transform dictionaries into a data_frame
    final_data_frame = pd.DataFrame()
    for unique in unique_classes:
        row = {'class_name': unique, 'hp': hp_dictionary[unique], 'hr': hr_dictionary[unique], 'hf': hf_dictionary[unique]}
        final_data_frame.append(row, ignore_index=True)

    return final_data_frame

def calculate_all_folds_conf_matrix(conf_matrix_list, size):
    final_conf_matrix = np.empty([size, size])

    for cm in conf_matrix_list:
        final_conf_matrix += cm

    return final_conf_matrix

def calculate_average_fold_result(fold_result_list, unique_classes):
    global_config = GlobalConfig.instance()
    strategy = fold_result_list[0].resampling_strategy
    algorithm = fold_result_list[0].resampling_algorithm

    # Retrieve list of results
    result_metrics_list = parse_result_metrics_list(fold_result_list)
    result_metrics_per_class = parse_result_metrics_per_class_list(fold_result_list)
    conf_matrix_list = parse_confusion_matrix_list(fold_result_list)

    # Calculating the metrics
    [avg_hp, avg_hr, avg_hf] = calculate_average_metrics(result_metrics_list)
    per_class_results_data_frame = calculate_average_metrics_per_class(result_metrics_per_class, unique_classes)
    final_cm = calculate_all_folds_conf_matrix(conf_matrix_list, len(unique_classes))

    # Plotting final_cm for the experiment
    image_path = global_config.directory_list['confusion_matrix_' + strategy + '_' + algorithm] + '/conf_matrix.png'
    plot_confusion_matrix(final_cm, classes=unique_classes, image_name=image_path,
                          normalize=True,
                          title='Confusion Matrix')

    average_experiment_results = ResultDTO(avg_hp, avg_hr, avg_hf)
    return [average_experiment_results, per_class_results_data_frame]