from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class ClassificationAlgorithm:

    def __init__(self, option):
        self.classifier_name = option
        if option == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=150, criterion='gini')
        elif option == 'mlp':
            self.classifier = MLPClassifier(solver='lbfgs', activation='relu', max_iter=500)
        elif option == 'svm':
            self.classifier = SVC(gamma='auto', probability=True)
        elif option == 'dt':
            self.classifier = DecisionTreeClassifier(criterion = 'gini')
        elif option == 'NB':
            self.classifier = GaussianNB()
        elif option == 'knn':
            self.classifier = KNeighborsClassifier(n_neighbors=5)

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
