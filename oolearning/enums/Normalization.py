from enum import unique, Enum, auto


@unique
class Normalization(Enum):
    NONE = auto()
    NORMALIZE = auto()
    CENTER_SCALE = auto()
    LOG_TRANSFORM = auto()
