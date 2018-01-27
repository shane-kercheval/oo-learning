from enum import unique, Enum, auto


@unique
class Skewness(Enum):
    NONE = auto()
    BOX_COX = auto()
