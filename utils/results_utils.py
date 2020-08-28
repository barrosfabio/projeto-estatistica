import csv
from matplotlib import pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix


"""
    This method writes a data_frame to CSV
"""
def write_csv(file_name, data_frame):
    csv_file_path = file_name + '.csv'
    header = list(data_frame.columns.values)

    print('Saving file to path: {}'.format(csv_file_path))

    with open(csv_file_path, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL, dialect='excel')

        # Write the header of the table
        filewriter.writerow(header)

        # Write all the other rows
        for index, row in data_frame.iterrows():
            filewriter.writerow(row)

"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(cm, classes, image_name,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          figsize=(20,20)):

    plt.figure(figsize=figsize)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(image_name)
    #plt.show(block=False)
    plt.close()


def build_and_plot_conf_matrix(image_path, output_array, predicted_array, normalize=False):

    conf_matrix_lbl = np.unique(predicted_array)
    conf_matrix = confusion_matrix(output_array, predicted_array)
    conf_matrix_path = image_path + '.png'

    plot_confusion_matrix(conf_matrix, classes=conf_matrix_lbl, image_name=conf_matrix_path,
                          normalize=normalize,
                          title='Confusion Matrix')

def parse_result_metrics_list(fold_result_list):
    result_metrics_list = []

    for result in fold_result_list:
        result_metrics_list.append(result.result_metrics)
    return result_metrics_list

def parse_result_metrics_per_class_list(fold_result_list):
    per_class_metrics_list = []
    for result in fold_result_list:
        per_class_metrics_list.append(result.result_metrics)


def calculate_average_fold_result(fold_result_list):
    result_metrics_list = parse_result_metrics_list(fold_result_list)
    [avg_hp, avg_hr, avg_hf] = calculate_average_metrics(result_metrics_list)




def calculate_average_metrics(result_metrics_list):
    hp = []
    hr = []
    hf = []

    for result_metrics in result_metrics_list:
        hp.append(result_metrics.hp)
        hr.append(result_metrics.hr)
        hf.append(result_metrics.hf)

    return [np.mean(hp), np.mean(hr), np.mean(hf)]

def calculate_average_metrics_per_class(fold):
    pass