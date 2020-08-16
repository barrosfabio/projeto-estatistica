import pandas as pd
from PIL import Image
import imghdr
from sklearn.model_selection import train_test_split


# Function to load an image from a path
def open_img(filename):
    img = Image.open(filename)
    return img


# Verify if a given image is using a valid format
def verify_valid_img(path):
    possible_formats = ['png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif']
    if imghdr.what(path) in possible_formats:
        return True
    else:
        return False

# Columns of the data_frame
def create_columns(column_number, property):
    columns = []
    for i in range(0, column_number):
        columns.append(str(i))

    columns.append(property)
    return columns


def slice_and_split_data_holdout(input_data, output_data, test_percentage):
    # Splitting the dataset in training/test using the Holdout technique
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(input_data, output_data,
                                                                              test_size=test_percentage,
                                                                              random_state=42)
    train_data_frame = pd.DataFrame(inputs_train)
    train_data_frame['class'] = outputs_train

    test_data_frame = pd.DataFrame(inputs_test)
    test_data_frame['class'] = outputs_test

    return [train_data_frame, test_data_frame]  # Return train and test data separately

# Function to split inputs and outputs
def slice_data(dataset):
    # Slicing the input and output data
    input_data = dataset.iloc[:, :-1].values
    output_data = dataset.iloc[:, -1].values

    return [input_data, output_data]



