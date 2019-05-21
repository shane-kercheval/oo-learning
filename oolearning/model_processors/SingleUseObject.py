import copy
from abc import ABCMeta, abstractmethod

from oolearning.model_wrappers.ModelExceptions import AlreadyExecutedError, NotExecutedError


class Cloneable(metaclass=ABCMeta):
    def clone(self):
        return copy.deepcopy(self)


class SingleUseObjectMixin(Cloneable):

    def __init__(self,
                 already_executed_exception_class=None,
                 not_executed_exception_class=None):
        self._has_executed = False
        if already_executed_exception_class is None:
            self._already_executed_exception_class = AlreadyExecutedError
        else:
            self._already_executed_exception_class = already_executed_exception_class

        if not_executed_exception_class is None:
            self._not_executed_exception_class = NotExecutedError
        else:
            self._not_executed_exception_class = not_executed_exception_class

    @property
    def has_executed(self):
        return self._has_executed

    def execute(self, **kwargs) -> object:
        self.ensure_has_not_executed()
        returned_object = self._execute(**kwargs)
        self._has_executed = True

        return returned_object

    @abstractmethod
    def _execute(self, **kwargs) -> object:
        pass

    @abstractmethod
    def additional_cloning_checks(self):
        """
        This method is so that child classes can define additional object state checks before cloning
            (e.g. see ModelWrapperBase which should not clone if the model-caching manager has already been
            set)
        """
        pass

    def clone(self):
        self.ensure_has_not_executed()
        self.additional_cloning_checks()
        return copy.deepcopy(self)

    def ensure_has_executed(self):
        if not self._has_executed:
            raise self._not_executed_exception_class()

    def ensure_has_not_executed(self):
        if self._has_executed:
            raise self._already_executed_exception_class()


