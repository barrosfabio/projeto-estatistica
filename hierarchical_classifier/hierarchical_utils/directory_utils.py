import calendar
import time
import os
from hierarchical_classifier.configurations.global_config import GlobalConfig
from hierarchical_classifier.constants.resampling_constants import NONE


def create_result_directories(result_path, resampling_strategies, resampling_algorithms):
    timestamp = calendar.timegm(time.gmtime())
    result_path = result_path + '_' + str(timestamp)
    if not os.path.isdir(result_path):
        print('Created directory {}'.format(result_path))
        os.mkdir(result_path)

    # List of directories and sub-directories where the results will be saved
    # This is the basic list
    directory_list = {'result': result_path,
                      'global': result_path + '/global',
                      'per_pipeline': result_path + '/per_pipeline',
                      'per_class': result_path + '/per_pipeline/per_class',
                      'confusion_matrix': result_path + '/per_pipeline/confusion_matrix',
                      'distribution': result_path + '/data_distribution'}

    # Adding directories for each resampling strategy
    for strategy in resampling_strategies:
        directory_list['per_class_'+strategy] = directory_list['per_class'] + '/' + strategy
        directory_list['confusion_matrix_' + strategy] = directory_list['confusion_matrix'] + '/' + strategy
        directory_list['distribution_' + strategy] = directory_list['distribution'] + '/' + strategy

    # Adding directories for each sampling algorithm
    for strategy in resampling_strategies:
        if strategy == NONE:
            algorithm = 'none'
            directory_list['per_class_' + strategy + '_' + algorithm] = directory_list['per_class_' + strategy] + '/' + algorithm
            directory_list['confusion_matrix_' + strategy + '_' + algorithm] = directory_list['confusion_matrix_' + strategy] + '/' + algorithm
            directory_list['distribution_' + strategy + '_' + algorithm] = directory_list['distribution_' + strategy] + '/' + algorithm
        else:
            for algorithm in resampling_algorithms:
                directory_list['per_class_' + strategy + '_' + algorithm] = directory_list['per_class_' + strategy] + '/' + algorithm
                directory_list['confusion_matrix_' + strategy + '_' + algorithm] = directory_list['confusion_matrix_' + strategy] + '/' + algorithm
                directory_list['distribution_' + strategy + '_' + algorithm] = directory_list['distribution_' + strategy] + '/' + algorithm

    for key, value in directory_list.items():
        if not os.path.isdir(value):
            print('Created directory {}'.format(value))
            os.mkdir(value)

    global_config = GlobalConfig.instance()
    global_config.set_directory_configuration(directory_list)