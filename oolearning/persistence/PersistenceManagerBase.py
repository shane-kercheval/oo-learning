from abc import ABCMeta, abstractmethod
from typing import Callable

from oolearning.model_processors.SingleUseObject import Cloneable


class PersistenceManagerBase(Cloneable, metaclass=ABCMeta):
    """
    # TODO Document
    """
    @abstractmethod
    def get_object(self, fetch_function: Callable[[], object], key: str=None) -> object:
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
