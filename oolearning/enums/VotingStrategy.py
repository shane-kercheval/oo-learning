from enum import unique, Enum, auto


@unique
class VotingStrategy(Enum):
    SOFT = auto()
    HARD = auto()
