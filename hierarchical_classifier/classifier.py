from utils.data_utils import *
from hierarchical_classifier.tree.lcpn_tree import LCPNTree

# Steps to build a hierarchical classifier

# Variables
train_path = '../feature_extraction/result/covid_feature_matrix_train.csv'
test_path = '../feature_extraction/result/covid_feature_matrix_test.csv'

# 1. Load the data from a CSV file
# 2. Get inputs and outputs
[train_data_frame, unique_train_classes] = load_csv_data(train_path)
[test_data_frame, unique_test_classes] = load_csv_data(test_path)


# 3. From the outputs array, use it to build the class_tree and to get the positive and negative classes according to a policy
tree = LCPNTree(unique_train_classes)
class_tree = tree.build_tree()

# 4. From the class_tree, retrieve the data for each node, based on the list of positive and negative classes
tree.retrieve_lcpn_data(class_tree, train_data_frame)

# 5. Train the classifiers


# 6. Predict


# 7. Calculate the results