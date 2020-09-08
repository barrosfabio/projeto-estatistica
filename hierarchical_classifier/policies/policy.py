from utils.class_relationship_utils import compare_child_length
from hierarchical_classifier.constants.utils_constants import CLASS_SEPARATOR
import numpy as np


class Policy:

    def __init__(self, current_class, parent_class):
        self.current_class = current_class
        self.parent_class = parent_class
        self.positive_classes = []
        self.negative_classes = []
        self.direct_child_classes = []

    def find_direct_child(self, combinations):
        combinations = list(combinations)
        immediate_child = {value for value in combinations if
                           value.find(self.current_class + CLASS_SEPARATOR) != -1 and compare_child_length(self.current_class,
                                                                                                       value)}
        self.direct_child_classes = np.array(list(immediate_child))
