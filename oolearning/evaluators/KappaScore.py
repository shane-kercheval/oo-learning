from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase
from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class KappaScore(UtilityFunctionMixin, ScoreActualPredictedBase):
    def __init__(self, converter: ContinuousToClassConverterBase):
        super().__init__()
        self._converter = converter

    @property
    def name(self) -> str:
        return Metric.KAPPA.value

    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:
        predicted_classes = self._converter.convert(values=predicted_values)

        return cohen_kappa_score(y1=actual_values, y2=predicted_classes)
