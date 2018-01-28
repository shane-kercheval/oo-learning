from typing import Tuple

import numpy as np

from oolearning.evaluators.EvaluatorBase import EvaluatorBase


# TODO: not implemented not tested
class MultiClassEvaluator(EvaluatorBase):

    @property
    def metric_name(self) -> str:
        return 'Multi-class Evaluator'

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        raise NotImplementedError()
