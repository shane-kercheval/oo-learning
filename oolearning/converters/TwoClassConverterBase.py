from typing import Union

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase


# noinspection PyAbstractClass
class TwoClassConverterBase(ContinuousToClassConverterBase):
    """
    Base class for Converters that support Two-class Classification problems and, therefore, have the concept
        of a `positive` class.
    """

    def __init__(self, positive_class: Union[str, int]):
        """
        :param positive_class: the value of the positive class of the target variable in the corresponding
            data-set being trained/predicted/evaluated.

            For example, for the titanic data-set, the positive class might be `lived`, or might be `1`
        """
        self._positive_class = positive_class

    @property
    def positive_class(self) -> Union[str, int]:
        """
        :return: the value associated with a positive class
        """
        return self._positive_class
