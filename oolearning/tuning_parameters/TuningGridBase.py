class TuningGridBase:
    def __init__(self, parameters: dict):
        self._tuning_params = parameters

    def _check_tuning_values(self):
        pass

    @property
    def grid(self):
        """
        logic to take parameters and find all combinations/permutations
        :return:
        """
        pass

