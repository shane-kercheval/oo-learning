from abc import ABCMeta, abstractmethod
from typing import Callable


class PersistenceManagerBase(metaclass=ABCMeta):
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
