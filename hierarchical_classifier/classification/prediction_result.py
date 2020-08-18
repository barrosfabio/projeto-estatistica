class PredictionResult:

    def __init__(self, predicted_class, probability, leaf_node_predicted):
        self.predicted_class = predicted_class
        self.probability = probability
        self.leaf_node_predicted = leaf_node_predicted
