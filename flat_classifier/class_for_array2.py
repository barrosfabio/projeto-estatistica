import numpy as np
import csv
from datetime import datetime

# For each folder
class Metrics:
    def __init__(self):
        self.accuracy_array = []
        self.precision_array = []
        self.recall_array = []


# For each Resample Method
class Resample:

    def __init__(self, resample_name):
        self.resample_name = resample_name
        self.metric = []

    def create_new_fold_metrics(self):
        self.metric.append(Metrics())

    def append_metrics_to_fold(self, fold, recall, accuracy, precision):
        self.metric[fold].recall_array.append(recall)
        self.metric[fold].accuracy_array.append(accuracy)
        self.metric[fold].precision_array.append(precision)

    def get_recall_average_from_fold(self, fold):
        return np.mean(self.metric[fold].recall_array)

    def get_detailed_recall_from_fold(self, fold):
        string = ''
        for value in self.metric[fold].recall_array:
            string += ', ' + str(value)

        return string

    def get_detailed_accuracy_from_fold(self, fold):
        string = ''
        for value in self.metric[fold].accuracy_array:
            string += ', ' + str(value)

        return string

    def get_detailed_precision_from_fold(self, fold):
        string = ''
        for value in self.metric[fold].precision_array:
            string += ', ' + str(value)

        return string

    def get_accuracy_average_from_fold(self, fold):
        return np.mean(self.metric[fold].accuracy_array)

    def get_precision_average_from_fold(self, fold):
        return np.mean(self.metric[fold].precision_array)


class AllResamples:

    def __init__(self):
        self.resample = [Resample('TESTE')]

    def create_new_resample(self, resample_name):
        self.resample.append(Resample(resample_name))

    def save_to_csv(self):
        filename = './flat_classifier_' + str(datetime.now().strftime('%M%S')) + '.csv'

        with open(filename, mode='w') as csv_file:
            fieldnames = ['resample_name', 'fold' ,'recall_average', 'accuracy_average', 'precision_average']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in self.resample:
                print('Saving data from: ', res.resample_name)

                for i in range(len(res.metric)):
                    print('- Fold: ', res.metric[i])

                    writer.writerow({
                        'resample_name': str(res.resample_name),
                        'fold': str(i),
                        'recall_average': str(res.get_recall_average_from_fold(i)),
                        'accuracy_average': str(res.get_recall_average_from_fold(i)),
                        'precision_average': str(res.get_recall_average_from_fold(i))})

    def save_detailed_to_csv(self):
        filename = './flat_detailed_lassifier_' + str(datetime.now().strftime('%M%S')) + '.csv'

        with open(filename, mode='w') as csv_file:
            fieldnames = ['resample_name', 'fold', 'recall_average', 'recall_datailed', 'accuracy_average',
                          'accuracy_datailed', 'precision_average', 'precision_datailed']

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in self.resample:
                print('Saving data from: ', res.resample_name)

                for i in range(len(res.metric)):
                    print('- Fold: ', res.metric[i])

                    writer.writerow({
                        'resample_name': str(res.resample_name),
                        'fold': str(i),
                        'recall_average': str(res.get_recall_average_from_fold(i)),
                        'recall_datailed': str(res.get_detailed_recall_from_fold(i)),
                        'accuracy_average': str(res.get_recall_average_from_fold(i)),
                        'accuracy_datailed': str(res.get_detailed_accuracy_from_fold(i)),
                        'precision_average': str(res.get_recall_average_from_fold(i)),
                        'precision_datailed': str(res.get_detailed_precision_from_fold(i)),})
