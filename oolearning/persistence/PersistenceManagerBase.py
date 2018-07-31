import copy

from abc import ABCMeta, abstractmethod
from typing import Callable

from oolearning.OOLearningHelpers import Lock


class PersistenceManagerBase(metaclass=ABCMeta):
    """
    # TODO Document
    """
    def clone(self):
        return copy.deepcopy(self)

    def get_object(self, fetch_function: Callable[[], object], key: str=None) -> object:
        try:
            Lock().acquire()
            return self._get_object(fetch_function=fetch_function, key=key)
        finally:
            Lock().release()

    @abstractmethod
    def _get_object(self, fetch_function: Callable[[], object], key: str = None) -> object:
        pass

    @abstractmethod
    def set_key(self, key: str):
        pass

    @abstractmethod
    def set_key_prefix(self, prefix: str):
        pass

    @abstractmethod
    def set_sub_structure(self, sub_structure: str):
        """
        :param sub_structure: For example, a subdirectory for a PersistenceManager based on a file system.
        """
        pass
