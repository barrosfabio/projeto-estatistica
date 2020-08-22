from hierarchical_classifier.results.results_framework import ResultsFramework

class FinalResultsFramework(ResultsFramework):

    def __init__(self):
        self.final_result_path = '/experiment_result/local_result'
        self.per_class_results = []
        self.per_parent_metrics = []
        super(FinalResultsFramework, self).__init__(self.local_result_path)