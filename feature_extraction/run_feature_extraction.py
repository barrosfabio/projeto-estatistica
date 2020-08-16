import numpy as np
import os
from feature_extraction.extractor import create_feature_matrix
from utils.data_utils import slice_data, slice_and_split_data_holdout

# Input parameters
# path of the images
images_directory = '../images'

# Type of extractor
lbp_extractor = 'nri_uniform'

# Setting up Result path
feature_matrix_path = os.getcwd()
feature_matrix_path = feature_matrix_path + '/result'

if not os.path.isdir(feature_matrix_path):
    os.mkdir(feature_matrix_path)

# percentage to be used in the holdout.
test_percentage = 0.4

print("Started to extract features...")
feature_matrix = create_feature_matrix(images_directory, lbp_extractor)
print("Saving Original Feature Matrix to CSV")
feature_matrix.to_csv(feature_matrix_path + '/covid_feature_matrix.csv', index=False)

print("Splitting Training and Test datasets using holdout...")
input_data, output_data= slice_data(feature_matrix)

[train_data_frame, test_data_frame] = slice_and_split_data_holdout(input_data, output_data, test_percentage)

print("Saving Both Feature Matrices to " + feature_matrix_path)

train_data_frame.to_csv(feature_matrix_path + '/covid_feature_matrix_train.csv', index=False)
print("Saving Train Feature Matrix to CSV Completed.")
test_data_frame.to_csv(feature_matrix_path + '/covid_feature_matrix_test.csv', index=False)
print("Saving Test Feature Matrix to CSV Completed.")


