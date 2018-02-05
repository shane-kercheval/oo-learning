# noinspection PyMethodMayBeStatic
class UtilityFunctionMixin:
    """
    UtilityFunctionMixin defines the better_than function for Evaluators where higher numbers are "better"
        (e.g. kappa, sensitivity, etc.)
    """
    def _better_than(self, this: float, other: float) -> bool:
        return this > other
