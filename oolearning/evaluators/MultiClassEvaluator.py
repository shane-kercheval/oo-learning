from typing import Tuple
import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class MultiClassEvaluator(EvaluatorBase):
    @property
    def metric_name(self) -> str:
        pass

    def _calculate_accuracy(self) -> Tuple[float, object]:
        pass
