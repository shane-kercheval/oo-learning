import copy
from abc import ABCMeta, abstractmethod


class ScoreBase(metaclass=ABCMeta):
    """
    A `Score` object is responsible for a single calculation representing the performance of a model. This
        type of object is used in Resamplers, Tuners, Searchers, etc., in order to track and compare the
        performance of various models.

    A Score (and the underlying the metric/calculation) can be a `utility`-function (higher scores are better)
        or a `cost`-function (lower scores are better).
    """
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
        """
        :return: the value of the metric/Score (set in `.calculate()`)
        """
        return self._value

    def calculate(self, *args) -> float:
        """
        given the actual and predicted values, this function calculates the corresponding value/score
        :param args: information necessary to calculate the score
        :return: calculated score
        """
        assert self._value is None  # we don't want to be able to reuse test_evaluators
        self._value = self._calculate(args)
        assert isinstance(self._value, float) or isinstance(self._value, int)
        return self._value

    def better_than(self, other: 'ScoreBase') -> bool:
        """
        used by the `__lt__()` function, not meant to be called by the user; only meant to serve as a wrapper
            around `_better_than()` which is meant to be overridden to define if the score is a
            `utility` or `cost` function.
        """
        assert isinstance(other, ScoreBase)
        return self._better_than(this=self.value, other=other.value)

    def __lt__(self, other: 'ScoreBase'):
        """
        :param other: another Score object
        :return:
        """
        return self.better_than(other=other)

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
        """
        compares `this` object's value to another Score's value, and determines which value is
            "better". If the Score is a cost function, then lower values are better; if the Score is a
            utility function, higher scores are better. Most Scores will utilize either `CostFunctionMixin` or
            `UtilityFunctionMixin` rather than overriding directly.
        """
        pass

    @abstractmethod
    def _calculate(self, *args) -> float:
        """
        This method calculates the value of the metric/Score.
        :param args: information necessary to calculate the score
        :return: the Score's value
        """
        pass


