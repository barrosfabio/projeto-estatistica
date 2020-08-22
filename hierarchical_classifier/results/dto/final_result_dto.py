from hierarchical_classifier.results.dto.result_dto import ResultDTO

class FinalResultDTO(ResultDTO):

    def __init__(self, hp, hr, hf, resampling_algorithm, resampling_strategy):
        self.resampling_algorithm = resampling_algorithm
        self.resampling_strategy = resampling_strategy
        super(FinalResultDTO, self).__init__(hp, hr, hf)