import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
import os
import seaborn as sns

class ResultsFramework:

    def __init__(self, results_path):
        self.results_path = results_path


    """
        This method prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
    """
    def plot_confusion_matrix(self, conf_matrix, classes, image_name,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              figsize=(20, 20)):

        image_path = self.results_path + image_name

        plt.figure(figsize=figsize)

        if normalize:
            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

        plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = conf_matrix.max() / 2.
        for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(image_path)
        # plt.show(block=False)
        plt.close()

    """
        This method plots a bar chart with the class distribution
    """
    def plot_class_dist(self, dict_values, image_name):

        data_dist_path = self.results_path + 'data_distribution'
        if not os.path.isdir(data_dist_path):
            os.mkdir(data_dist_path)
        data_dist_path = data_dist_path + image_name + '.png'

        x = np.empty(0)
        y = np.empty(0)
        i = 0;
        for key, value in dict_values.items():
            x = np.append(x, key)
            y = np.append(y, value)
            i += 1

        plt.figure(figsize=(45, 45))
        sns.barplot(x=x, y=y)
        plt.xticks(rotation=90)
        plt.savefig(data_dist_path)
        # plt.show()
        plt.cla()
        plt.close()

