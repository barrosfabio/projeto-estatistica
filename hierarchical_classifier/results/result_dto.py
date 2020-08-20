class ResultDTO:

    def __init__(self, hp, hr, hf, resampling_algorithm, resampling_strategy):
        self.hp = hp
        self.hr = hr
        self.hf = hf
        self.resampling_algorithm = resampling_algorithm
        self.resampling_strategy = resampling_strategy