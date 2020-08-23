import numpy as np
import matplotlib.pyplot as plt
import operator
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_data(path):
    # Loading the CSV Data
    data_frame = pd.read_csv(path)
    classes = data_frame['class']

    # Gathering the unique classes
    unique_classes = np.unique(classes)

    return [data_frame, unique_classes]


def count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(classes)):
        print('Class {}, Count {}'.format(classes[i], count[i]))
    print('')

def count_by_class(output_values):
    label_count = np.unique(output_values, return_counts=True)
    key_count_dict = {}
    genres = label_count[0]
    counts = label_count[1]
    count = pd.DataFrame()

    for i in range(0, len(genres)):
        key_count_dict[genres[i]] = counts[i]

    sorted_dict = dict(sorted(key_count_dict.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_dict.items():
        row = {'class': key, 'count': value}
        count = count.append(row, ignore_index=True)

    return count

def slice_and_split_data_holdout(input_data, output_data, test_percentage):
    print('Original class distribution')
    count_per_class(output_data)
    # TODO: Print the original class distribution here
    # Splitting the dataset in training/test using the Holdout technique
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(input_data, output_data,
                                                                              test_size=test_percentage,
                                                                              random_state=42)

    train_data_frame = pd.DataFrame(inputs_train)
    train_data_frame['class'] = outputs_train
    print('Training dataset class distribution')
    count_per_class(outputs_train)

    test_data_frame = pd.DataFrame(inputs_test)
    test_data_frame['class'] = outputs_test
    print('Testing dataset class distribution')
    count_per_class(outputs_test)

    return [train_data_frame, test_data_frame]  # Return train and test data separately

# Function to split inputs and outputs
def slice_data(dataset):
    # Slicing the input and output data
    input_data = dataset.iloc[:, :-1].values
    output_data = dataset.iloc[:, -1].values

    return [input_data, output_data]
