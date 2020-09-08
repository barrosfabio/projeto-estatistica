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


data = './feature_extraction/result/filtered_covid_canada_rydles_plus7.csv'
classifier = 'rf'
resample = False

resample_metrics = AllResamples()

# Load data
data_frame = pd.read_csv(data)

kfold = StratifiedKFold(n_splits=5, shuffle=True)


[input_data, output_data] = some_functions.slice_data(data_frame)
clf = some_functions.get_classifier(classifier)
classes = np.unique(output_data)
print(len(classes))

for res, resampling_algorithm_name in enumerate(resampling_algorithm.resampling_algorithms):
    kfold_count = 1

    resample_metrics.create_new_resample(resampling_algorithm_name)

    all_folds_outputs_test = []
    all_folds_predicted = []

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
            resampler = resampling_algorithm.instantiate_resampler('flat', resampling_algorithm_name, 5)

            [input_data_train, output_data_train] = resampler.fit_resample(inputs_train,
                                                                           outputs_train)  # Original class distribution
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

        # precision, recall, f1 = hierarchical_metrics.calculate_hierarchical_metrics(predicted, outputs_train)
        f1 = f1_score(outputs_test, predicted, average='macro')
        precision = precision_score(outputs_test, predicted, average='macro')
        recall = recall_score(outputs_test, predicted, average='macro')


        resample_metrics.resample[res].create_new_fold_metrics()
        resample_metrics.resample[res].append_metrics_to_fold(kfold_count - 1, recall, f1, precision)
        resample_metrics.resample[res].print_averages_from_fold(kfold_count - 1)
        # ...print_detailed_averages_from_fold(kfold_count-1)

        all_folds_outputs_test.extend(outputs_test)
        all_folds_predicted.extend(predicted)

        print('=' * 50)
        print('')

        kfold_count += 1

    conf_matrix = confusion_matrix(all_folds_outputs_test, all_folds_predicted)
    plot_confusion_matrix(conf_matrix, classes=classes, image_name=f'./conf_matrix_flat_{resampling_algorithm_name}_normalized.png',
                          normalize=True,
                          title='Confusion Matrix')
    plot_confusion_matrix(conf_matrix, classes=classes, image_name=f'./conf_matrix_flat_{resampling_algorithm_name}.png',
                          normalize=False,
                          title='Confusion Matrix')

resample_metrics.save_to_csv()
resample_metrics.save_detailed_to_csv()

print(len(classes))