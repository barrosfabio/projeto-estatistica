from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_flat_metrics(predicted_classes, outputs_test, avg_type):
    precision = precision_score(outputs_test, predicted_classes, average=avg_type)
    recall = recall_score(outputs_test, predicted_classes, average=avg_type)
    f1score = f1_score(outputs_test, predicted_classes, average=avg_type)

    return [precision, recall, f1score]