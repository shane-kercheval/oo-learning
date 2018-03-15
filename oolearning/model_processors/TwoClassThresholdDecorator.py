from typing import List

import numpy as np

from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.converters.TwoClassRocOptimizerConverter import TwoClassRocOptimizerConverter
from oolearning.converters.TwoClassPrecisionRecallOptimizerConverter import TwoClassPrecisionRecallOptimizerConverter  # noqa


class TwoClassThresholdDecorator(DecoratorBase):
    def __init__(self):
        super().__init__()
        self._resampled_roc = list()
        self._resampled_precision_recall = list()

    # noinspection PyProtectedMember
    def decorate(self, **kwargs):
        # Specific to 2-class classification; need to use the right objects, or this will explode.
        # in future, if there will be additional shit like this, could consider passing in a list
        # of objects with common interface that is ran at the end of each resample fold
        # and iterating through objects, which hold the specific information (like resampled
        # thresholds)
        scores = kwargs['scores']
        holdout_actual_values = kwargs['holdout_actual_values']
        holdout_predicted_values = kwargs['holdout_predicted_values']

        positive_class = None
        # score should either have `_positive_class` directly (e.g. AUC) or in the converter
        first_score = scores[0]
        if hasattr(first_score, '_positive_class'):
            positive_class = first_score.positive_class
        elif hasattr(first_score, '_converter'):
            if hasattr(first_score._converter, 'positive_class'):
                positive_class = first_score._converter.positive_class

        if positive_class is None:
            raise ValueError("Cannot find positive class in Score or Score's Converter")

        converter = TwoClassRocOptimizerConverter(actual_classes=holdout_actual_values,
                                                  positive_class=positive_class)
        converter.convert(values=holdout_predicted_values)
        self._resampled_roc.append(converter.ideal_threshold)

        converter = TwoClassPrecisionRecallOptimizerConverter(actual_classes=holdout_actual_values,
                                                              positive_class=positive_class)
        converter.convert(values=holdout_predicted_values)
        self._resampled_precision_recall.append(converter.ideal_threshold)

    @property
    def resampled_roc(self) -> List[float]:
        return self._resampled_roc

    @property
    def resampled_precision_recall(self) -> List[float]:
        return self._resampled_precision_recall

    @property
    def resampled_roc_mean(self) -> float:
        return float(np.mean(self._resampled_roc))

    @property
    def resampled_precision_recall_mean(self) -> float:
        return float(np.mean(self.resampled_precision_recall))

    @property
    def resampled_roc_st_dev(self) -> float:
        return float(np.std(self._resampled_roc))

    @property
    def resampled_precision_recall_st_dev(self) -> float:
        return float(np.std(self.resampled_precision_recall))

    @property
    # coefficient of variation
    def resampled_roc_cv(self):
        return round((self.resampled_roc_st_dev / self.resampled_roc_mean), 2)

    @property
    # coefficient of variation
    def resampled_precision_recall_cv(self) -> float:
        return round((self.resampled_precision_recall_st_dev / self.resampled_precision_recall_mean), 2)
