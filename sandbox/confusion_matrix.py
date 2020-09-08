from sklearn.metrics import confusion_matrix
from utils.results_utils import plot_confusion_matrix
import numpy as np

def build_conf_matrix():

    output_classes_1 = ['R/1', 'R/1', 'R/1', 'R/2', 'R/3', 'R/2', 'R/3', 'R/3', 'R/1', 'R/2']
    predicted_classes_1 = ['R/1', 'R/1', 'R/2', 'R/2', 'R/2', 'R/2', 'R/3', 'R/1', 'R/1', 'R/2']

    output_classes_2 = ['R/2', 'R/1', 'R/1', 'R/2', 'R/3', 'R/2', 'R/3', 'R/2', 'R/1', 'R/2', 'R/1']
    predicted_classes_2 = ['R/2', 'R/1', 'R/1', 'R/2', 'R/3', 'R/2', 'R/3', 'R/1', 'R/1', 'R/2', 'R/1']

    cm1 = confusion_matrix(output_classes_1, predicted_classes_1)
    cm2 = confusion_matrix(output_classes_2, predicted_classes_2)

    cm_list = [cm1, cm2, cm1, cm2, cm2]

    return cm_list







unique_classes = ['R/1', 'R/2', 'R/3']
image_path = '/hierarchical_classifier/results'
conf_matrix_path = image_path + '/conf_matrix.png'
cm_list = build_conf_matrix()
final_cm = np.empty([len(unique_classes), len(unique_classes)])
for cm in cm_list:
    final_cm+=cm


plot_confusion_matrix(final_cm, classes=unique_classes, image_name=conf_matrix_path,
                      normalize=True,
                      title='Confusion Matrix')