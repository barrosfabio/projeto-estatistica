from sklearn.ensemble import RandomForestClassifier


class ClassificationAlgorithm():

    def __init__(self, option):
        if option == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=150, criterion='gini')

    def train(self, dataset):
        # Training the classifier
        self.classifier = self.classifier.fit(dataset.inputs, dataset.outputs)

    def prediction(self, data):
        data = data.reshape(1, -1)
        predicted_class = self.classifier.predict(data)

        return predicted_class

    def prediction_proba(self, data):
        data = data.reshape(1, -1)
        predicted_class = self.classifier.predict(data)
        predicted_proba = self.classifier.predict_proba(data)

        return [predicted_class, predicted_proba]
