import numpy as np
import csv
from datetime import datetime


# For each folder
class Metrics:
    def __init__(self):
        self.HF = []
        self.HP = []
        self.HR = []


# For each Resample Method
class Resample:

    def __init__(self, resample_name):
        self.resample_name = resample_name
        self.metric = []

    def create_new_fold_metrics(self):
        self.metric.append(Metrics())

    def append_metrics_to_fold(self, fold, hr, hf, hp):
        self.metric[fold].HR.append(hr)
        self.metric[fold].HF.append(hf)
        self.metric[fold].HP.append(hp)

    # GET FROM FOLD
    def get_metric_average_from_fold(self, metric, fold):
        if metric == 'HR':
            return np.mean(self.metric[fold].HR)
        elif metric == 'HF':
            return np.mean(self.metric[fold].HF)
        elif metric == 'HP':
            return np.mean(self.metric[fold].HP)

    # GET DETAILED FROM FOLD
    def get_detailed_metric_from_fold(self, metric, fold):
        string = ''

        if metric == 'HR':
            for value in self.metric[fold].HR:
                string += ', ' + str(value)
            return string

        elif metric == 'HF':
            for value in self.metric[fold].HF:
                string += ', ' + str(value)
            return string

        elif metric == 'HP':
            for value in self.metric[fold].HP:
                string += ', ' + str(value)
            return string

    # FINAL AVERAGES
    def get_final_average(self, metric):
        final = []

        if metric == 'HR':
            for met in self.metric:
                final += met.HR

        elif metric == 'HF':
            for met in self.metric:
                final += met.HF

        elif metric == 'HP':
            for met in self.metric:
                final += met.HP

        return np.mean(final)

    # PRINT
    def print_averages_from_fold(self, fold):
        print('{:>50}'.format(str(f'Avg HR: {self.get_metric_average_from_fold("HR", fold):.5f}')))
        print('{:>50}'.format(str(f'Avg HF: {self.get_metric_average_from_fold("HF", fold):.5f}')))
        print('{:>50}'.format(str(f'Avg HP: {self.get_metric_average_from_fold("HP", fold):.5f}')))

    def print_detailed_averages_from_fold(self, fold):
        # HR
        print('-' * 50)
        print('HR values from fold {}: '.format(fold))
        for rc in self.metric[fold].recall_array:
            print(rc, end=' -> ')
        print('END')
        print('{:>50}'.format(str(f'Avg HR: {np.mean(self.metric[fold].HR):.5f}')))

        # HF
        print('-' * 50)
        print('HF values from fold {}: '.format(fold))
        for ac in self.metric[fold].accuracy_array:
            print(ac, end=' -> ')
        print('END')
        print('{:>50}'.format(str(f'Avg HF: {np.mean(self.metric[fold].HF):.5f}')))

        # HP
        print('-' * 50)
        print('HP values from fold {}: '.format(fold))
        for pr in self.metric[fold].precision_array:
            print(pr, end=' -> ')
        print('END')
        print('{:>50}'.format(str(f'Avg HP: {np.mean(self.metric[fold].HP):.5f}')))


class AllResamples:

    def __init__(self):
        self.resample = []

    def create_new_resample(self, resample_name):
        self.resample.append(Resample(resample_name))

    def save_to_csv(self, filename):
        print('\nStarted saving detailed data process.')

        with open(filename, mode='w') as csv_file:
            fieldnames = ['resample_name', 'fold', 'HR_average', 'HF_average', 'HP_average']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in self.resample:
                print('- Saving data from: {:>15}'.format(str(res.resample_name)), end=' - ')
                print('Fold: ', end=' ')

                for i in range(len(res.metric)):
                    print(i, end=' .. ')

                    writer.writerow({
                        'resample_name': str(res.resample_name),
                        'fold': str(i),
                        'HR_average': str(res.get_metric_average_from_fold('HR', i)),
                        'HF_average': str(res.get_metric_average_from_fold('HF', i)),
                        'HP_average': str(res.get_metric_average_from_fold('HP', i))})

                writer.writerow({
                    'resample_name': str(res.resample_name),
                    'fold': str('TOTAL AVERAGE'),
                    'HR_average': str(res.get_final_average('HR')),
                    'HF_average': str(res.get_final_average('HF')),
                    'HP_average': str(res.get_final_average('HP'))})

                print('END')

        print('Simple data save complete!')
        print('Created file -> ', filename)

    def save_detailed_to_csv(self, filename):
        print('\nStarted saving detailed data process.')

        with open(filename, mode='w') as csv_file:
            fieldnames = ['resample_name', 'fold', 'HR_average', 'HR_datailed', 'HF_average',
                          'HF_datailed', 'HP_average', 'HP_datailed']

            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            for res in self.resample:
                print('- Saving data from: {:>15}'.format(str(res.resample_name)), end=' - ')
                print('Fold: ', end=' ')

                for i in range(len(res.metric)):
                    print(i, end=' .. ')

                    writer.writerow({
                        'resample_name': str(res.resample_name),
                        'fold': str(i),
                        'HR_average': str(res.get_metric_average_from_fold('HR', i)),
                        'HR_datailed': str(res.get_detailed_metric_from_fold('HR', i)),
                        'HF_average': str(res.get_metric_average_from_fold('HF', i)),
                        'HF_datailed': str(res.get_detailed_metric_from_fold('HF', i)),
                        'HP_average': str(res.get_metric_average_from_fold('HP', i)),
                        'HP_datailed': str(res.get_detailed_metric_from_fold('HP', i)), })

                print('END')

        print('Detailed data save complete!')
        print('Created file -> ', filename)
