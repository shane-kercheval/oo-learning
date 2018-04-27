from typing import List

import numpy as np

from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.converters.TwoClassRocOptimizerConverter import TwoClassRocOptimizerConverter
from oolearning.converters.TwoClassPrecisionRecallOptimizerConverter import TwoClassPrecisionRecallOptimizerConverter  # noqa


class TwoClassThresholdDecorator(DecoratorBase):
    """
    In object-oriented programming, the `decorator` is pattern is described as a way to "attach additional
    responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing for
    extending functionality." (https://sourcemaking.com/design_patterns/decorator)

    This Decorator is passed into a resampler and, for each time the model is trained in the Resampler, the
        Decorator is run via `.decorate()`, calculating the ideal thresholds that minimizes the
        distance to the upper left corner for the ROC curve, and minimizes the distance to the upper right
        corner for the Precision/Recall curve (i.e. balancing the inherent trade-offs in both curves.
    """
    def __init__(self):
        super().__init__()
        self._roc_ideal_thresholds = list()
        self._precision_recall_ideal_thresholds = list()

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
        # we need to get the name of the 'positive class;
        # the Score object should either have `_positive_class` directly (e.g. AUC) or in the converter
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
        self._roc_ideal_thresholds.append(converter.ideal_threshold)  # TODO: rename roc_ideal_thresholds to something like resampled_roc_ideal_threshold ... shitty name.

        converter = TwoClassPrecisionRecallOptimizerConverter(actual_classes=holdout_actual_values,
                                                              positive_class=positive_class)
        converter.convert(values=holdout_predicted_values)
        self._precision_recall_ideal_thresholds.append(converter.ideal_threshold)

    @property
    def roc_ideal_thresholds(self) -> List[float]:
        """
        :return: for each time the model is trained in the Resampler, the threshold associated with the
            point on the ROC curve that minimizes the distance to the upper left corner of the curve (thereby
            balancing the trade-off between the true positive rate (sensitivity) and the true negative rate
            (specificity)) is added to the list of "ideal thresholds", which is returned.
        """
        return self._roc_ideal_thresholds

    @property
    def precision_recall_ideal_thresholds(self) -> List[float]:
        """
        :return: each time the model is trained in the Resampler, the threshold associated with the
            point on the precision/recall curve that minimizes the distance to the upper right corner of the
            curve (thereby balancing the trade-off between the true positive rate (recall) and the positive
            predictive value (precision)) is added to the list of "ideal thresholds", which is returned.
        """
        return self._precision_recall_ideal_thresholds

    @property
    def roc_ideal_thresholds_mean(self) -> float:
        """
        :return: the mean of the "ideal" thresholds
        """
        return float(np.mean(self._roc_ideal_thresholds))

    @property
    def resampled_precision_recall_mean(self) -> float:
        """
        :return: the mean of the "ideal" thresholds
        """
        return float(np.mean(self.precision_recall_ideal_thresholds))

    @property
    def roc_ideal_thresholds_st_dev(self) -> float:
        """
        :return: the standard deviation of the "ideal" thresholds
        """
        return float(np.std(self._roc_ideal_thresholds))

    @property
    def resampled_precision_recall_st_dev(self) -> float:
        """
        :return: the standard deviation of the "ideal" thresholds
        """
        return float(np.std(self.precision_recall_ideal_thresholds))

    @property
    def roc_ideal_thresholds_cv(self):
        """
        :return: the coefficient of variation of the "ideal" thresholds
        """
        return round((self.roc_ideal_thresholds_st_dev / self.roc_ideal_thresholds_mean), 2)

    @property
    def resampled_precision_recall_cv(self) -> float:
        """
        :return: the coefficient of variation of the "ideal" thresholds
        """
        return round((self.resampled_precision_recall_st_dev / self.resampled_precision_recall_mean), 2)
