import numpy as np
import os
from feature_extraction.feature_extraction_utils.feature_extraction_utils import *
from feature_extraction.lbp.lbp import LocalBinaryPatterns
import pandas as pd

UNIFORM_FEATURE_NUMBER = 10
NRI_UNIFORM_FEATURE_NUMBER = 59


# Feature extraction call
def feature_extraction(image, lbp_extractor):
    lbp = LocalBinaryPatterns(8, 2)
    image_matrix = np.array(image.convert('L'))

    if lbp_extractor == 'uniform':
        img_features = lbp.describe_lbp_method_rd(image_matrix)
    elif lbp_extractor == 'nri_uniform':
        img_features = lbp.describe_lbp_method_ag(image_matrix)

    return img_features.tolist()


# Function to create the training feature matrix, it has the expected class for each sample
def create_feature_matrix(images_directory, extractor_type):
    # Variable to store the data_rows
    rows_list = []

    print("Started feature extraction for the training dataset")

    # Iterate over subdirectories in training folder (1 folder for each class)
    for directory in os.listdir(images_directory):

        # This is the path to each subdirectory
        sub_directory = images_directory + '/' + directory

        # Retrieve the files for the given subdirectory
        training_filelist = os.listdir(sub_directory)

        # The name of the directory is the class
        class_name = 'R/' + directory

        # Iterate over all the files in the class folder
        for file in training_filelist:
            file_path = sub_directory + '/' + file

            if verify_valid_img(file_path):
                _fileNumber = str(training_filelist.index(file) + 1) + " from " + str(len(training_filelist))
                print("Processing: " + _fileNumber + " -- PATH: " + file_path)

                image = open_img(file_path)
                img_features = feature_extraction(image, extractor_type)

                # Replacing underscores with slashes
                class_name = class_name.replace("_", "/")

                img_features.append(class_name)

                rows_list.append(img_features)
            else:
                print("The following file is not a valid image: " + file_path)

        print("-" * 100)
    # Creating a dataframe to store all the features
    if extractor_type == 'uniform':
        columns = create_columns(UNIFORM_FEATURE_NUMBER, 'class')
    elif extractor_type == 'nri_uniform':
        columns = create_columns(NRI_UNIFORM_FEATURE_NUMBER, 'class')

    feature_matrix = pd.DataFrame(rows_list, columns=columns)

    print("Finished creating Training Feature Matrix")

    return feature_matrix
