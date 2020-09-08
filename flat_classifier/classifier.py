import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from flat_classifier import some_functions, resampling_algorithm

# Options
from flat_classifier.class_for_array import AllResamples

data = './feature_extraction/result/filtered_covid_canada_rydles_plus7.csv'
classifier = 'rf'
resample = False
accuracy_array = []
f1_array = []
precision_array = []
recall_array = []

"""
test_resample = [Resample('SMOTE'), Resample('TESTE')]
test_resample[0].create_new_fold_metrics()
test_resample[0].append_metrics_to_fold(0, 2, 4, 6)
test_resample[0].append_metrics_to_fold(0, 4, 4, 6)
test_resample[0].append_metrics_to_fold(0, 2, 2, 6)

test_resample[0].print_averages_from_fold(0)

print(test_resample[0].print_detailed_averages_from_fold(0))
"""

resample_metrics = AllResamples()
resample_metrics.create_new_resample('SVM')
resample_metrics.resample[0].create_new_fold_metrics()
resample_metrics.resample[0].append_metrics_to_fold(0, 4, 5, 7)
resample_metrics.resample[0].append_metrics_to_fold(0, 6, 5, 7)
resample_metrics.resample[0].append_metrics_to_fold(0, 6, 4, 7)

resample_metrics.resample[1].create_new_fold_metrics()
resample_metrics.resample[1].append_metrics_to_fold(0, 3, 5, 7)
resample_metrics.resample[1].append_metrics_to_fold(0, 4, 5, 8)
resample_metrics.resample[1].append_metrics_to_fold(0, 5, 5, 8)


resample_metrics.save_to_csv()
resample_metrics.save_detailed_to_csv()

# Load data
data_frame = pd.read_csv(data)

kfold = StratifiedKFold(n_splits=5, shuffle=True)


""""""


for resampling_algorithm_name in resampling_algorithm.resampling_algorithms:

    clf = some_functions.get_classifier(classifier)

    [input_data, output_data] = some_functions.slice_data(data_frame)

    kfold_count = 1

    for train_index, test_index in kfold.split(input_data, output_data):
        a = ' Started fold ' + str(kfold_count) + ' '
        print('{:~^50}'.format(a))

        # Slice inputs and outputs
        inputs_train, outputs_train = input_data[train_index], output_data[train_index]
        inputs_test, outputs_test = input_data[test_index], output_data[test_index]

        # Original class distribution
        some_functions.count_per_class(outputs_train)
        # if count < 3: continue

        # If resample flag is True, we need to resample the training dataset by generating new synthetic samples
        if resampling_algorithm_name:
                
            print(f"Resampling data with {resampling_algorithm_name}")
            resampler = resampling_algorithm.instantiate_resampler('flat', resampling_algorithm_name, 3)

            [input_data_train, output_data_train] = resampler.fit_resample(inputs_train, outputs_train)  # Original class distribution
            print("Done resampling")

            # Resampled class distribution
            some_functions.count_per_class(output_data_train)

        # Train the classifier
        print("Started training ", end='')
        clf = clf.fit(inputs_train, outputs_train)
        print("-> Finished")

        # Predict
        print("Start Prediction ", end='')
        predicted = clf.predict(inputs_test)
        print("-> Finished")

        a = ' Results for fold ' + str(kfold_count) + ' '
        print('{:-^50}'.format(a))

        f1 = f1_score(outputs_test, predicted, average='micro')
        recall = recall_score(outputs_test, predicted, average='micro')  # micro, macro, weighted
        precision = precision_score(outputs_test, predicted, average='micro')

        recall_array.append(recall)
        f1_array.append(f1)
        precision_array.append(precision)

        print('Recall MICRO: ', str(recall))
        print('F1 Score: ' + str(f1))
        print('Precision MICRO: ', str(precision))

        print('=' * 50)
        print('')

        kfold_count += 1

    print('{:-^50}'.format(' FINAL RESULTS '))
    print('Avg Recall: {}'.format(np.mean(recall_array)))
    print('Avg F1: {}'.format(np.mean(f1_array)))
    print('Avg Precision: {}'.format(np.mean(precision_array)))
