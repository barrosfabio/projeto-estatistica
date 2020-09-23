from hierarchical_classifier.constants.utils_constants import CLASS_SEPARATOR

def find_ancestors(predicted, actual, current_pos):
    ancestors_predicted = predicted[0:current_pos]
    ancestors_actual = actual[0:current_pos]

    count_equal = 0
    for i in range(0, current_pos):
        if ancestors_predicted[i] == ancestors_actual[i]:
            count_equal += 1

    if count_equal == len(ancestors_predicted) and count_equal == len(ancestors_actual):
        return True
    else:
        return False


def calculate_intersection(predicted, actual):
    if len(predicted) < len(actual):
        smallest = len(predicted)
    elif len(predicted) == len(actual):
        smallest = len(predicted)
    elif len(predicted) > len(actual):
        smallest = len(actual)

    intersection = 0
    for i in range(0, smallest):
        ## Increases the intersection only if current position is equal and the ancestors in common
        if predicted[i] == actual[i] and find_ancestors(predicted, actual, i):
            intersection += 1

    return intersection


def hierarchical_precision(predicted, actual):
    sum_intersections = 0
    sum_predicted = 0

    for i in range(len(predicted)):
        prediction_splitted = predicted[i].split(CLASS_SEPARATOR)
        actual_splitted = actual[i].split(CLASS_SEPARATOR)

        # Calculate Intersection of the
        intersection = calculate_intersection(prediction_splitted[1:], actual_splitted[1:])
        sum_intersections += intersection
        sum_predicted += (len(prediction_splitted) - 1)

    print('Sum predicted: {}'.format(sum_predicted))
    print('Sum intersections: {}'.format(sum_intersections))

    hp = sum_intersections / sum_predicted
    return hp


def hierarchical_recall(predicted, actual):
    sum_intersections = 0
    sum_actual = 0

    for i in range(len(predicted)):
        prediction_splitted = predicted[i].split(CLASS_SEPARATOR)
        actual_splitted = actual[i].split(CLASS_SEPARATOR)

        intersection = calculate_intersection(prediction_splitted[1:], actual_splitted[1:])
        sum_intersections += intersection
        sum_actual += (len(actual_splitted) - 1)
    hr = sum_intersections / sum_actual

    print('Sum actual: {}'.format(sum_actual))
    print('Sum intersections: {}'.format(sum_intersections))
    return hr


def hierarchical_fmeasure(hierarchical_precision, hierarchical_recall):
    if hierarchical_precision == 0 or hierarchical_recall == 0:
        return 0.0
    else:
        hf1 = (2 * hierarchical_precision * hierarchical_recall) / (hierarchical_precision + hierarchical_recall)
        return hf1


def calculate_hierarchical_metrics(predicted_classes, outputs_test):
    hp = hierarchical_precision(predicted_classes, outputs_test)
    hr = hierarchical_recall(predicted_classes, outputs_test)
    hf = hierarchical_fmeasure(hp, hr)

    return [hp, hr, hf]