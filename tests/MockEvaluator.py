from typing import Tuple

import numpy as np

from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


# noinspection PyAbstractClass
class MockEvaluator(EvaluatorBase):
    def __init__(self, metric_name):
        super().__init__()
        self._metric_name = metric_name

    @property
    def metric_name(self) -> str:
        return self._metric_name

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        pass


class MockUtilityEvaluator(UtilityFunctionMixin, MockEvaluator):
    pass


class MockCostEvaluator(CostFunctionMixin, MockEvaluator):
    pass
