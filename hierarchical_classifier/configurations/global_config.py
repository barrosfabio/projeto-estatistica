class GlobalConfig():

    _instance = None

    def __init__(self):
        self.results_dir = None
        self.global_results_dir = None
        self.local_results_dir = None
        self.data_distribution_dir = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_directory_configuration(self, directory_list):
        self.results_dir = directory_list['result']
        self.global_results_dir = directory_list['global']
        self.local_results_dir = directory_list['local']
        self.data_distribution_dir = directory_list['distribution']
        self.hierarchical_data_dist = directory_list['hierarchical_local_dist']


