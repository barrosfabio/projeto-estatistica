from sklearn.model_selection import StratifiedKFold
from utils.data_utils import slice_data, load_csv_data, count_per_class

path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_canada_feature_matrix.csv'
folds = 5

# Load the data from a CSV file
[data_frame, unique_classes] = load_csv_data(path)
[input_data, output_data] = slice_data(data_frame)

kfold = StratifiedKFold(n_splits=folds, shuffle=True)
kfold_count = 1
split = kfold.split(input_data, output_data)

for train_index, test_index in split:
    print('\n------------- Started fold {} -------------'.format(kfold_count))
    print('Training folds have {} samples'.format(len(train_index)))
    input_data_train, output_data_train = input_data[train_index], output_data[train_index]
    print('Training dataset distribution')
    count_per_class(output_data_train)
    print('Train Indexes')
    print(train_index)

    print('\nTest fold has {} samples'.format(len(test_index)))
    input_data_test, output_data_test = input_data[test_index], output_data[test_index]
    print('Test dataset distribution')
    count_per_class(output_data_test)
    print('\nTest Indexes')
    print(test_index)
    kfold_count +=1
