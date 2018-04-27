import copy

from abc import ABCMeta, abstractmethod


class DecoratorBase(metaclass=ABCMeta):
    """
    In object-oriented programming, the `decorator` is pattern is described as a way to "attach additional
        responsibilities to an object dynamically. Decorators provide a flexible alternative to subclassing
        for extending functionality." (https://sourcemaking.com/design_patterns/decorator)

    For example, to piggy-back off of the Resampler and do a calculation or capture data at the end of each
        fold.
    """
    @abstractmethod
    def decorate(self, **kwargs):
        pass

    def clone(self):
        return copy.deepcopy(self)
