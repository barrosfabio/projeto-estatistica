from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
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
        return MLPClassifier(solver='lbfgs', activation='relu', max_iter=500)
    elif classifier == 'svm':
        return SVC(gamma='auto', probability=True)
    elif classifier == 'dt':
        return DecisionTreeClassifier(criterion='gini')
    elif classifier == 'nb':
        return GaussianNB()
    elif classifier == 'knn':
        return KNeighborsClassifier(n_neighbors=5)
