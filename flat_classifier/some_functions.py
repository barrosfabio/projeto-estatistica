from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np


# Slice inputs and outputs
def slice_data(dataset):
    # Slicing the input and output data
    input_data = dataset.iloc[:, :-1].values
    output_data = dataset.iloc[:, -1].values

    return [input_data, output_data]


def print_count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(original_count) + 1):
        print('Class {}, Count {}'.format(classes[i], count[i]))
    print('')



def get_classifier(classifier):
    # Define the classifier
    if classifier == 'rf':
        return RandomForestClassifier(n_estimators=150, criterion='gini')
    elif classifier == 'mlp':
        return MLPClassifier(hidden_layer_sizes=120, activation='logistic', verbose=False, early_stopping=True,
                             validation_fraction=0.2)
    elif classifier == 'svm':
        return SVC(gamma='auto', probability=True)
