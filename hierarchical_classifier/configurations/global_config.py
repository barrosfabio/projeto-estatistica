class GlobalConfig():

    _instance = None

    def __init__(self):
        self.directory_list = None
        self.k_neighbors = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_directory_configuration(self, directory_list):
        self.directory_list = directory_list

    def set_kneighbors(self, k_neighbors):
        self.k_neighbors = k_neighbors

    def set_kfold(self, kfold):
        self.kfold = kfold
