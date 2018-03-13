from enum import unique, Enum


@unique
class DummyClassifierStrategy(Enum):
    STRATIFIED = 'stratified'
    MOST_FREQUENT = 'most_frequent'
    PRIOR = 'prior'
    UNIFORM = 'uniform'
