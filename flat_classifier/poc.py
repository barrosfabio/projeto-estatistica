import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flat_classifier import some_functions, resampling_algorithm
from hierarchical_classifier.evaluation import hierarchical_metrics
# Options
from flat_classifier.class_for_array import AllResamples
from sklearn.metrics import confusion_matrix
from utils.results_utils import plot_confusion_matrix

from datetime import datetime

from hierarchical_classifier.configurations.global_config import GlobalConfig
from hierarchical_classifier.hierarchical_utils.directory_utils import create_result_directories
from hierarchical_classifier.constants.resampling_constants import *

from hierarchical_classifier.results.pipeline_results_framework import PipeleineResultsFramework
from hierarchical_classifier.evaluation.flat_metrics import calculate_flat_metrics
from hierarchical_classifier.results.dto.experiment_result_dto import ExperimentResultDTO
from hierarchical_classifier.results.dto.result_dto import ResultDTO
from hierarchical_classifier.hierarchical_utils.hierarchical_results_utils import calculate_average_fold_result
from hierarchical_classifier.results.dto.average_experiment_result_dto import AverageExperimentResultDTO
from hierarchical_classifier.results.global_results_framework import GlobalResultsFramework


data = './feature_extraction/result/filtered_covid_canada_rydles_plus7.csv'
data_frame = pd.read_csv(data)

[input_data, output_data] = some_functions.slice_data(data_frame)
classes = np.unique(output_data)

print(classes)
print(np.unique(output_data, return_counts=True))

predicted = ['R/Normal'] * len(output_data)

[hp, hr, hf] = calculate_flat_metrics(predicted, output_data, avg_type='macro')

print(hp)
print(hr)
print(hf)