from hierarchical_classifier.results.results_framework import ResultsFramework
import pandas as pd
import os

class GlobalResultsFramework(ResultsFramework):
    global_results_header = ['resampling_strategy','resampling_algorithm','hp','hr','hf']

    def transform_to_csv(self, result_list):
        path = self.results_path + '/global_results'

        if not os.path.isdir(path):
            os.mkdir(path)

        file_name = path + '/experiment_results'

        result_data_frame = pd.DataFrame(columns=self.global_results_header)

        for result in result_list:
            row = {'resampling_strategy':result.resampling_strategy,'resampling_algorithm':result.resampling_algorithm,'hp':result.hp,'hr':result.hr,'hf':result.hf}

            result_data_frame = result_data_frame.append(row, ignore_index=True)

        self.write_csv(file_name, result_data_frame)