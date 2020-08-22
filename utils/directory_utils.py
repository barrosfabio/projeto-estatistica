import calendar
import time
import os
from hierarchical_classifier.configurations.global_config import GlobalConfig


def create_result_directories(result_path):
    timestamp = calendar.timegm(time.gmtime())
    result_path = result_path + '_' + str(timestamp)

    # List of directories and sub-directories where the results will be saved
    directory_list = {'result': result_path,
                      'global': result_path + '/global',
                      'local': result_path + '/local',
                      'distribution': result_path + '/data_distribution'}

    for key, value in directory_list.items():
        if not os.path.isdir(value):
            print('Created directory {}'.format(value))
            os.mkdir(value)

    global_config = GlobalConfig.instance()
    global_config.set_directory_configuration(directory_list)