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
        self.per_class_results = []
        self.per_parent_metrics = []
        super(GlobalResultsFramework, self).__init__(global_result_path)

    def transform_to_csv(self, result_list):
        path = self.results_path

        if not os.path.isdir(path):
            os.mkdir(path)

        file_name = path + '/experiment_results'

        result_data_frame = pd.DataFrame(columns=self.global_results_header)

        for result in result_list:
            row = {'resampling_strategy':result.resampling_strategy,'resampling_algorithm':result.resampling_algorithm,'hp':result.hp,'hr':result.hr,'hf':result.hf}

            result_data_frame = result_data_frame.append(row, ignore_index=True)

        write_csv(file_name, result_data_frame)