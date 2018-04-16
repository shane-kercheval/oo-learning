from typing import Callable

from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase


class AlwaysFetchManager(PersistenceManagerBase):
    """
    'fetches' i.e. gets/creates the object every time i.e. no persistence
    """
    def get_object(self, fetch_function: Callable[[], object], key: str=None) -> object:
        return fetch_function()

    def set_key(self, key: str):
        pass

    def set_key_prefix(self, prefix: str):
        pass

    def set_sub_structure(self, sub_structure: str):
        pass
