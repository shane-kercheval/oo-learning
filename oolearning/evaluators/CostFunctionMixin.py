# noinspection PyMethodMayBeStatic
class CostFunctionMixin:
    """
    CostFunctionMixin defines the better_than function for Evaluators where lower numbers are "better"
        (e.g. RMSE, Error Rate, etc.)
    """
    def _better_than(self, this: float, other: float) -> bool:
        return this < other
