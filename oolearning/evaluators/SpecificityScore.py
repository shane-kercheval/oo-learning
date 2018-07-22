from typing import Union

import numpy as np
import pandas as pd

from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase
from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase
from oolearning.enums.Metric import Metric
from oolearning.evaluators.TwoClassConfusionMatrix import TwoClassConfusionMatrix
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class SpecificityScore(UtilityFunctionMixin, ScoreActualPredictedBase):
    def __init__(self, converter: TwoClassConverterBase):
        super().__init__()
        self._converter = converter

    @property
    def name(self) -> str:
        return Metric.SPECIFICITY.value

    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:
        predicted_classes = self._converter.convert(values=predicted_values)

        return TwoClassConfusionMatrix(actual_classes=actual_values,
                                       predicted_classes=predicted_classes,
                                       positive_class=self._converter.positive_class).true_negative_rate
