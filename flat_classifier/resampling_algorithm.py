from imblearn.combine import SMOTETomek, SMOTEENN
from hierarchical_classifier.constants.resampling_constants import *
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN, KMeansSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, TomekLinks, EditedNearestNeighbours, \
    NearMiss, AllKNN

resampling_algorithms = [NONE, RANDOM_OVERSAMPLER, SMOTE_RESAMPLE, BORDERLINE_SMOTE, ADASYN_RESAMPLER, SMOTE_ENN, SMOTE_TOMEK]
resampling_strategies = [NONE, FLAT_RESAMPLING]

# Resampling strategies
NONE = 'none'
FLAT_RESAMPLING = 'flat'
HIERARCHICAL_RESAMPLING = 'hierarchical'

# Resampling algorithms
RANDOM_OVERSAMPLER = 'ros'
SMOTE_RESAMPLE = 'smote'
BORDERLINE_SMOTE = 'borderline'
KMEANS_SMOTE = 'kmeans'
SVM_SMOTE = 'svm'
ADASYN_RESAMPLER = 'adasyn'
SMOTE_ENN = 'smote-enn'
SMOTE_TOMEK = 'smote-tomek'
RANDOM_UNDERSAMPLER = 'rus'
NEIGHBOUR_CLEANING = 'ncr'
TOMEK = 'tomek'
ENN = 'enn'
NEAR_MISS = 'near_miss'
ALL_KNN = 'all-knn'





def instantiate_resampler(self, algorithm_name, k_neighbors):
    n_jobs = 1
    sampling_strategy = 'auto'
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42, n_jobs=n_jobs)

    if algorithm_name == SMOTE_RESAMPLE:
        return smote
    elif algorithm_name == SMOTE_ENN:
        return SMOTEENN(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs, smote=smote)
    elif algorithm_name == SMOTE_TOMEK:
        return SMOTETomek(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs, smote=smote)
    elif algorithm_name == BORDERLINE_SMOTE:
        return BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42, n_jobs=n_jobs)
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
