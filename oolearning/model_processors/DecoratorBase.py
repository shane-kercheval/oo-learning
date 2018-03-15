from abc import ABCMeta, abstractmethod


class DecoratorBase(metaclass=ABCMeta):
    """
    intent is to add responsibility objects dynamically
    For example, to piggy-back off of the Resampler and do a calculation or capture data at the end of each
    fold.
    """
    @abstractmethod
    def decorate(self, **kwargs):
        pass
