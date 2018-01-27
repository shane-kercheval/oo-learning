from enum import unique, Enum, auto


@unique
class ResolveOutliers(Enum):
    NONE = auto()
    SPATIAL_SIGN = auto()
    REMOVE_1_5_IQR = auto()
    REMOVE_TOP_BOTTOM_1_PERCENT = auto()
