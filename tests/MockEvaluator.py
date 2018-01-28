from typing import Tuple

import numpy as np

from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class MockEvaluator(EvaluatorBase):
    def __init__(self, metric_name, better_than):
        super().__init__(better_than=better_than)
        self._metric_name = metric_name

    @property
    def metric_name(self) -> str:
        return self._metric_name

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        pass
