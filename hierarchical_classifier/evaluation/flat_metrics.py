from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_flat_metrics(predicted_classes, outputs_test):
    precision = precision_score(outputs_test, predicted_classes, average='macro')
    recall = recall_score(outputs_test, predicted_classes, average='macro')
    f1score = f1_score(outputs_test, predicted_classes, average='macro')

    return [precision, recall, f1score]