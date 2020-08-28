from hierarchical_classifier.results.dto.result_dto import ResultDTO

class FinalResultDTO(ResultDTO):

    def __init__(self, result_metrics, per_class_metrics, resampling_algorithm, resampling_strategy, fold):
        self.result_metrics = result_metrics
        self.per_class_metrics = per_class_metrics
        self.resampling_algorithm = resampling_algorithm
        self.resampling_strategy = resampling_strategy
        self.fold = fold
        super(FinalResultDTO, self).__init__(result_metrics.hp, result_metrics.hr, result_metrics.hf)