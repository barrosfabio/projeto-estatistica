import pandas as pd
import os

nome = 'SVM'
file_directory = '../flat_classifier/' + nome
columns = ['none', 'ros', 'smote', 'borderline', 'adasyn', 'smote-enn', 'smote-tomek']
final_df = pd.DataFrame()
final_file_name = '../flat_classifier/GlobalResultsMerge/results_' + nome + '.csv'

dir_list = os.listdir(file_directory)

for dir in dir_list:
    file_path = file_directory + '\\' + dir + '\\global\\experiment_results.csv'
    print(file_path)
    data_frame = pd.read_csv(file_path, sep=';')
    transposed_data_frame = data_frame.transpose()

    final_df = final_df.append(transposed_data_frame)

final_df = final_df.filter(like='hf', axis=0)
final_df.to_csv(final_file_name, sep=';')
