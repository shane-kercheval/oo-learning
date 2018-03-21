from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase


class MockHyperParams(HyperParamsBase):
    def __init__(self):
        super().__init__()
        self._params_dict = {'criterion': None,
                             'max_features': None,
                             'n_estimators': None,
                             'min_samples_leaf': None}  # have to set, because base won't set non-existent key
