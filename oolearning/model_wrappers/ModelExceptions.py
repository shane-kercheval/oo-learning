class ModelNotFittedError(Exception):
    pass


class ModelAlreadyFittedError(Exception):
    pass


class ModelCachedAlreadyConfigured(Exception):
    pass


class MissingValueError(Exception):
    pass


class NegativeValuesFoundError(Exception):
    pass


class AlreadyExecutedError(Exception):
    pass


class NotExecutedError(Exception):
    pass
