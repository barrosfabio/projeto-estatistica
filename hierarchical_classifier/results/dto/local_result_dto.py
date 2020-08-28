from hierarchical_classifier.results.dto.result_dto import ResultDTO

class LocalResultDTO(ResultDTO):

    def __init__(self, hp, hr, hf, class_name, resampling_strategy, resampling_):
        self.class_name = class_name
        self.resampling_strategy = resampling_strategy
        self
        super(LocalResultDTO, self).__init__(hp, hr, hf)