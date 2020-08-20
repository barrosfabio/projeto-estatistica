from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN
from utils.data_utils import slice_data
import pandas as pd
from hierarchical_classifier.constants.resampling_constants import *


class ResamplingAlgorithm:

    def __init__(self, algorithm_name, k_neighbors):
        self.algorithm_name = algorithm_name

        if algorithm_name == SMOTE_RESAMPLE:
            self.resampler = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42, n_jobs=4)
        elif algorithm_name == SMOTE_ENN:
            smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42, n_jobs=4)
            self.resampler = SMOTEENN(sampling_strategy='auto', random_state=42, n_jobs=4, smote=smote)
        elif algorithm_name == SMOTE_TOMEK:
            smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42, n_jobs=4)
            self.resampler = SMOTETomek(sampling_strategy='auto', random_state=42, n_jobs=4, smote=smote)
        elif algorithm_name == BORDERLINE_SMOTE:
            smote = SMOTE(sampling_strategy='auto', k_neighbors=k_neighbors, random_state=42, n_jobs=4)
            self.resampler = BorderlineSMOTE(sampling_strategy='auto', random_state=42, n_jobs=4, smote=smote)
        elif algorithm_name == ADASYN_RESAMPLER:
            self.resampler = ADASYN(sampling_strategy='auto', random_state=42, n_jobs=4, n_neighbors=k_neighbors)
        elif algorithm_name == RANDOM_OVERSAMPLER:
            self.resampler = RandomOverSampler(sampling_strategy='auto', random_state=42)

    # Executes resampling
    def resample(self, data_frame):
        [input_data, output_data] = slice_data(data_frame)
        [input_data, output_data] = self.resampler.fit_resample(input_data, output_data)

        resampled_data_frame = pd.DataFrame(input_data)
        resampled_data_frame['class'] = output_data
        return resampled_data_frame
