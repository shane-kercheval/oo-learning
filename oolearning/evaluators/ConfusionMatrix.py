import numpy as np
import pandas as pd


class ConfusionMatrix:
    def __init__(self, actual_classes: np.ndarray, predicted_classes: np.ndarray):
        self._matrix = pd.crosstab(actual_classes, predicted_classes,
                                   rownames=['actual'], colnames=['predictions'])

    @property
    def matrix(self):
        self._matrix

    def accuracy(self):
        pass








