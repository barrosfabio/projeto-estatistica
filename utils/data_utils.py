import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv_data(path):
    # Loading the CSV Data
    data_frame = pd.read_csv(path)
    classes = data_frame['class']

    # Gathering the unique classes
    unique_classes = np.unique(classes)

    # Slicing in Input and Output data
    [input_data, output_data] = slice_data(data_frame)

    return [input_data, output_data, unique_classes]


def count_per_class(output_data):
    original_count = np.unique(output_data, return_counts=True)
    classes = original_count[0]
    count = original_count[1]

    for i in range(0, len(classes)):
        print('Class {}, Count {}'.format(classes[i], count[i]))
    print('')

def plot_class_dist(dict_values, image_path,  image_name):

    path = image_path
    if not os.path.isdir(path):
        os.mkdir(path)
    path = path + '_' + image_name + '.png'

    x = np.empty(0)
    y = np.empty(0)
    i = 0;
    for key, value in dict_values.items():
        x = np.append(x, key)
        y = np.append(y, value)
        i += 1

    plt.figure(figsize=(45, 45))
    sns.barplot(x=x, y=y)
    plt.xticks(rotation=90)
    plt.savefig(path)
    plt.cla()
    plt.close()

def slice_and_split_data_holdout(input_data, output_data, test_percentage):
    print('Original class distribution')
    count_per_class(output_data)
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