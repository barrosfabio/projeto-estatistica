from hierarchical_classifier.results.dto.result_dto import ResultDTO

class LocalResultDTO(ResultDTO):

    def __init__(self, hp, hr, hf, class_name):
        self.class_name = class_name
        super(LocalResultDTO, self).__init__(hp, hr, hf)