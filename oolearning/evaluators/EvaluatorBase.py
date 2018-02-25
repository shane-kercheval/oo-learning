from abc import ABCMeta, abstractmethod

import numpy as np


class EvaluatorBase(metaclass=ABCMeta):
    @abstractmethod
    def evaluate(self, actual_values: np.ndarray, predicted_values: object):
        pass

    @property
    @abstractmethod
    def all_quality_metrics(self) -> dict:
        pass
