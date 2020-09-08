from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN, KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, TomekLinks, EditedNearestNeighbours, NearMiss, AllKNN
from utils.data_utils import slice_data
from hierarchical_classifier.constants.resampling_constants import *
from hierarchical_classifier.configurations.global_config import GlobalConfig
from utils.data_utils import count_by_class
from utils.results_utils import write_csv
import pandas as pd
import os


class ResamplingAlgorithm:

    def __init__(self, resampling_strategy, algorithm_name, k_neighbors, class_name=None):
        self.resampling_strategy = resampling_strategy
        self.algorithm_name = algorithm_name
        self.class_name = class_name
        self.resampler = self.instantiate_resampler(algorithm_name, k_neighbors)

    # Instantiates a resampling algorithm based on the parameter provided
    def instantiate_resampler(self, algorithm_name, k_neighbors):
        n_jobs = 4
        sampling_strategy = 'auto'
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42, n_jobs=n_jobs)

        if algorithm_name == SMOTE_RESAMPLE:
            return smote
        elif algorithm_name == SMOTE_ENN:
            return SMOTEENN(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs, smote=smote)
        elif algorithm_name == SMOTE_TOMEK:
            return SMOTETomek(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs, smote=smote)
        elif algorithm_name == BORDERLINE_SMOTE:
            return BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs, k_neighbors=k_neighbors)
        elif algorithm_name == ADASYN_RESAMPLER:
            return ADASYN(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs, n_neighbors=k_neighbors)
        elif algorithm_name == RANDOM_OVERSAMPLER:
            return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        elif algorithm_name == KMEANS_SMOTE:
            return KMeansSMOTE(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs)
        elif algorithm_name == SVM_SMOTE:
            return SVMSMOTE(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs, k_neighbors=k_neighbors)
        elif algorithm_name == RANDOM_UNDERSAMPLER:
            return RandomUnderSampler(sampling_strategy='majority')
        elif algorithm_name == NEIGHBOUR_CLEANING:
            return NeighbourhoodCleaningRule(sampling_strategy='majority', n_jobs=n_jobs)
        elif algorithm_name == TOMEK:
            return TomekLinks(sampling_strategy='majority', n_jobs=n_jobs)
        elif algorithm_name == ENN:
            return EditedNearestNeighbours(sampling_strategy='majority', n_jobs=n_jobs)
        elif algorithm_name == NEAR_MISS:
            return NearMiss(sampling_strategy='majority', n_jobs=n_jobs)
        elif algorithm_name == ALL_KNN:
            return AllKNN(sampling_strategy='majority', n_jobs=n_jobs)

    # Executes resampling
    def resample(self, data_frame, k_fold):

        [input_data, output_data] = slice_data(data_frame)
        before_resample = count_by_class(output_data)

        [input_data, output_data] = self.resampler.fit_resample(input_data, output_data)
        after_resample = count_by_class(output_data)

        self.save_class_distribution(before_resample, after_resample, k_fold)

        resampled_data_frame = pd.DataFrame(input_data)
        resampled_data_frame['class'] = output_data

        return resampled_data_frame

    # Saves the class distribution before and after resampling
    def save_class_distribution(self, before_resample, after_resample, k_fold):
        global_config = GlobalConfig.instance()
        data_dist_path = global_config.directory_list['distribution_' + self.resampling_strategy + '_' + self.algorithm_name]
        data_dist_path = data_dist_path + '/' + 'fold_' + str(k_fold)
        if not os.path.isdir(data_dist_path):
            os.mkdir(data_dist_path)

        if self.resampling_strategy == HIERARCHICAL_RESAMPLING:
            class_name  = self.class_name.replace('/','_')
            before_file_name = data_dist_path + '/before_resample_' + self.algorithm_name + '_' + class_name
            after_file_name = data_dist_path + '/after_resample_' + self.algorithm_name + '_' + class_name
        else:
            before_file_name = data_dist_path + '/before_resample_' + self.algorithm_name
            after_file_name = data_dist_path + '/after_resample_' + self.algorithm_name

        write_csv(before_file_name, before_resample)
        write_csv(after_file_name, after_resample)