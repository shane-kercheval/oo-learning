class CallbackUsedWithParallelizationError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Cannot use a callback with using parallelization')
