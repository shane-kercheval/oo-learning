from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase


# noinspection PyAbstractClass
class TwoClassConverterBase(ContinuousToClassConverterBase):
    def __init__(self, positive_class):
        self._positive_class = positive_class

    @property
    def positive_class(self):
        return self._positive_class
