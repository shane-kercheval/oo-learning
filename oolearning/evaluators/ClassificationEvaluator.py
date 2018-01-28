from abc import abstractmethod
from typing import Tuple, List, Callable
import numpy as np

from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class ClassificationEvaluator(EvaluatorBase):
    """
    Base class for TwoClassEvaluator & MultiClassEvaluator
    """
    def __init__(self,
                 better_than: Callable[[float, float], bool],
                 categories: List,
                 use_probabilities: bool=True,
                 threshold: float=0.50):
        super().__init__(better_than=better_than)
        self._categories = categories
        self._use_probabilities = use_probabilities
        self._threshold = threshold if use_probabilities else None
        self._custom_threshold = True if threshold is not None else False
        self._predicted_values = None
        self._actual_values = None

    @property
    @abstractmethod
    def metric_name(self) -> str:
        pass

    @abstractmethod
    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        pass
