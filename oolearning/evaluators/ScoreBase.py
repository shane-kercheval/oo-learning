import copy
from abc import ABCMeta, abstractmethod

import numpy as np

from oolearning import OOLearningHelpers
from oolearning.evaluators.OutcomeTypeNotSupportedError import OutcomeTypeNotSupportedError


# TODO TEST
class SupportsAnyClassificationMixin:
    # noinspection PyMethodMayBeStatic
    def _supports_values(self, values: np.ndarray):

        if OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype):
            raise OutcomeTypeNotSupportedError('Score does not support numeric values')


# TODO TEST
class SupportsTwoClassClassificationMixin:
    # noinspection PyMethodMayBeStatic
    def _supports_values(self, values: np.ndarray):
        if OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype):
            raise OutcomeTypeNotSupportedError('Score does not support numeric values')

        if len(set(values)) > 2:
            raise OutcomeTypeNotSupportedError('Score does not support multi-class values')


# TODO TEST
class SupportsTwoClassProbabilitiesMixin:
    # noinspection PyMethodMayBeStatic
    def _supports_values(self, values: np.ndarray):
        if not OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype):
            raise OutcomeTypeNotSupportedError('Score needs numeric probabilities')

        if any(values < 0) or any(values > 1):
            raise OutcomeTypeNotSupportedError('Score needs numeric probabilities between 0 and 1')


# TODO TEST
class SupportsMultiClassClassificationMixin:
    # noinspection PyMethodMayBeStatic
    def _supports_values(self, values: np.ndarray):
        if OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype):
            raise OutcomeTypeNotSupportedError('Score does not support numeric values')

        if len(set(values)) <= 2:
            raise OutcomeTypeNotSupportedError('Score only support multi-class values')


class SupportsRegressionMixin:
    # noinspection PyMethodMayBeStatic
    def _supports_values(self, values: np.ndarray):
        if not OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype):
            raise OutcomeTypeNotSupportedError('Score only supports numeric values')


class ScoreBase(metaclass=ABCMeta):
    def __init__(self):
        self._value = None

    def clone(self):
        """
        when, for example, resampling, an Evaluator will have to be cloned several times (before using)
        :return: a clone of the current object
        """
        assert self._value is None  # only intended on being called before evaluating
        return copy.deepcopy(self)

    @property
    def value(self) -> float:
        assert isinstance(self._value, float)
        return self._value

    def better_than(self, other: 'ScoreBase') -> bool:
        assert isinstance(other, ScoreBase)
        return self._better_than(this=self.value, other=other.value)

    def calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        """
        given the actual and predicted values, this function calculates the corresponding value/score
        :param actual_values: actual values
        :param predicted_values: predicted values (from a trained model)
        :return: calculated score
        """
# TODO TEST
        self._supports_values(values=actual_values)

        assert self._value is None  # we don't want to be able to reuse test_evaluators
        assert actual_values.shape == predicted_values.shape
        self._value = self._calculate(actual_values, predicted_values)
        assert isinstance(self._value, float)
        return self._value

    def __lt__(self, other):
        return self.better_than(other=other)

    # TODO: TEST
    @staticmethod
    def _is_regression(values: np.ndarray) -> bool:

        return OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype)

    # TODO: TEST
    @staticmethod
    def _is_two_class(values: np.ndarray) -> bool:
        return not OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype) and \
               len(set(values)) == 2

    # TODO: TEST
    @staticmethod
    def _is_multi_class(values: np.ndarray) -> bool:
        return not OOLearningHelpers.is_series_dtype_numeric(dtype=values.dtype) and \
               len(set(values)) > 2

    ##########################################################################################################
    # Abstract Methods
    ##########################################################################################################
    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: 'friendly' name to identify the metric such as 'RMSE'
        """
        pass

    @abstractmethod
    def _better_than(self, this: float, other: float) -> bool:
        pass

# TODO TEST
    @abstractmethod
    def _supports_values(self, values: np.ndarray):
        """
        :param values: values to check (e.g. actual values)
        :return: Nothing if Success, OutcomeTypeNotSupportedError with corresponding message
        """
        pass

    @abstractmethod
    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        pass


# noinspection PyAbstractClass
class ClassificationScoreBase(ScoreBase):
    def __init__(self, positive_class=None):
        """

        :param positive_class:
        """
        super().__init__()
        self._positive_class = positive_class

    # TODO TEST that i'm overriding and valueerror
    def calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        if self._is_two_class(values=actual_values) and self._positive_class is None:
            raise ValueError('positive_class needs to be set in order to use two-class scores')

        return super().calculate(actual_values=actual_values, predicted_values=predicted_values)
