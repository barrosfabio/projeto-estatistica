
import pandas as pd
from hierarchical_classifier.results.dto.local_result_dto import LocalResultDTO
import numpy as np

def build_list_df():
    row_1 = {'class_name':'R/1' , 'hp': 0.5 , 'hr':0.6 , 'hf':0.7}
    row_2 = {'class_name':'R/2' , 'hp': 0.6 , 'hr':0.7 , 'hf':0.8}
    row_3 = {'class_name':'R/3' , 'hp': 0.8 , 'hr':0.8 , 'hf':0.8}
    row_4 = {'class_name':'R/4' , 'hp': 0.8 , 'hr':0.9 , 'hf':0.8}

    data_frame = pd.DataFrame()

    data_frame = data_frame.append(row_1, ignore_index=True)
    data_frame = data_frame.append(row_2, ignore_index=True)
    data_frame = data_frame.append(row_3, ignore_index=True)
    data_frame = data_frame.append(row_4, ignore_index=True)

    return [data_frame, data_frame, data_frame, data_frame, data_frame]

def build_list_object():
    obj1 = LocalResultDTO(0.5, 0.6, 0.7, 'R/1')
    obj2 = LocalResultDTO(0.6, 0.7, 0.7, 'R/2')
    obj3 = LocalResultDTO(0.7, 0.75, 0.8, 'R/3')
    obj4 = LocalResultDTO(0.8, 0.8, 0.9, 'R/4')
    obj5 = LocalResultDTO(0.8, 0.78, 0.79, 'R/1')
    obj6 = LocalResultDTO(0.65, 0.71, 0.72, 'R/2')
    obj7 = LocalResultDTO(0.75, 0.72, 0.81, 'R/3')
    obj8 = LocalResultDTO(0.88, 0.89, 0.91, 'R/4')

    list = [obj1, obj2, obj3, obj4]
    list2 = [obj5, obj6, obj7, obj8]

    return [list, list2, list, list2, list]


list_results = build_list_object()

hp = 0.0
hr = 0.0
hf = 0.0

unique_classes  = ['R/1', 'R/2', 'R/3', 'R/4']
#hp_dictionary = {'R/1': [], 'R/2': [], 'R/3': [], 'R/4': []}
hp_dictionary = {}
hr_dictionary = {}
hf_dictionary = {}

first_fold = list_results[0]

# Initialize dictionaries
for unique in unique_classes:
    hp_dictionary[unique] = []
    hr_dictionary[unique] = []
    hf_dictionary[unique] = []

# Write results
for fold_result in list_results:
    for obj in fold_result:
        hp_dictionary[obj.class_name].append(obj.hp)
        hr_dictionary[obj.class_name].append(obj.hr)
        hf_dictionary[obj.class_name].append(obj.hf)

# Calculating avg for each class
for unique in unique_classes:
    hp_dictionary[unique] = np.mean(hp_dictionary[unique])
    hr_dictionary[unique] = np.mean(hr_dictionary[unique])
    hf_dictionary[unique] = np.mean(hf_dictionary[unique])

# Transform dictionaries into a data_frame
final_data_frame = pd.DataFrame()
for unique in unique_classes:
    row = {'class_name': unique, 'hp': hp_dictionary[unique], 'hr': hr_dictionary[unique], 'hf': hf_dictionary[unique]}
    final_data_frame = final_data_frame.append(row, ignore_index=True)

print(final_data_frame)