from enum import Enum, auto, unique


@unique
class CategoricalEncoding(Enum):
    NONE = auto()
    ONE_HOT = auto()
    DUMMY = auto()
