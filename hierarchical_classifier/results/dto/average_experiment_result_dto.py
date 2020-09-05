
class AverageExperimentResultDTO():

    def __init__(self, avg_result, resampling_strategy, resampling_algorithm):
        self.avg_result = avg_result
        self.resampling_strategy = resampling_strategy
        self.resampling_algorithm = resampling_algorithm