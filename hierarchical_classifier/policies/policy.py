from utils.class_relationship_utils import find_immediate_child

class Policy():

    def __init__(self, current_class, parent_class):
        self.current_class = current_class
        self.parent_class = parent_class
        self.positive_classes = []
        self.negative_classes = []
        self.direct_child_classes = []


    def find_direct_child(self, combinations):
        self.direct_child_classes = find_immediate_child(self.current_class, combinations)