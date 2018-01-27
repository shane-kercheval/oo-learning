from enum import unique, Enum, auto


@unique
class Imputation(Enum):
    NONE = auto()
    KNN = auto()
    MEDIAN = auto()
