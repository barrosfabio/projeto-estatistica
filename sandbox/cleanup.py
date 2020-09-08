import pandas as pd
import os.path
from os import path
import shutil
import numpy as np
import operator

def count_by_class(output_values):
    label_count = np.unique(output_values, return_counts=True)
    key_count_dict = {}
    genres = label_count[0]
    counts = label_count[1]
    count = pd.DataFrame()

    for i in range(0, len(genres)):
        key_count_dict[genres[i]] = counts[i]

    sorted_dict = dict(sorted(key_count_dict.items(), key=operator.itemgetter(1), reverse=True))
    return sorted_dict

#path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_canada_feature_matrix.csv'
#path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/covid_canada_rydles_feature_matrix.csv'
path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/filtered_covid_canada.csv'
#path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/filtered_covid_canada_rydles.csv'

new_path = 'C:/Users/Fabio Barros/Git/projeto-estatistica/feature_extraction/result/filtered_covid_canada_plus7.csv'

classes = ['R/Viral/Varicella','R/Viral/Influenza','R/Bacterial/Escherichia Coli','R/Bacterial/Chlamydophila','R/Bacterial/Unknown']

df = pd.read_csv(path)

filtered_df = df[~df['class'].isin(classes)]
finding = filtered_df['class']
dict = count_by_class(finding)

print('Dataset: {}'.format(path))
for key, value in dict.items():
    print('Class {}: {}'.format(key, value))
print('Total {} Samples'.format(len(df)))

filtered_df.to_csv(new_path, index=False)