from hierarchical_classifier.results.results_framework import ResultsFramework
from utils.results_utils import write_csv
from hierarchical_classifier.configurations.global_config import GlobalConfig
import pandas as pd
import os

class GlobalResultsFramework(ResultsFramework):
    global_results_header = ['resampling_strategy','resampling_algorithm','hp','hr','hf']

    def __init__(self):
        global_config = GlobalConfig.instance()
        global_result_path = global_config.directory_list['global']
        super(GlobalResultsFramework, self).__init__(global_result_path)

    def transform_to_csv(self, result_list, file_name):
        path = self.results_path

        if not os.path.isdir(path):
            os.mkdir(path)

        file_name = path + '/' + file_name

        result_data_frame = pd.DataFrame(columns=self.global_results_header)

        for result in result_list:
            row = {'resampling_strategy': result.resampling_strategy,
                       'resampling_algorithm': result.resampling_algorithm, 'hp': result.avg_result.hp,
                       'hr': result.avg_result.hr, 'hf': result.avg_result.hf}

            result_data_frame = result_data_frame.append(row, ignore_index=True)

        write_csv(file_name, result_data_frame)

    def transform_per_class_to_csv(self, per_class_list):
        global_config = GlobalConfig.instance()

        for per_class in per_class_list:
            strategy = per_class.resampling_strategy
            algorithm = per_class.resampling_algorithm
            data_frame = per_class.avg_result

            path = global_config.directory_list['per_class_' + strategy + '_' + algorithm]
            file_name = path + '/per_class_metrics_' + strategy + '_' + algorithm

            write_csv(file_name, data_frame)