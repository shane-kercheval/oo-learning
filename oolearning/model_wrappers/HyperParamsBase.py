from abc import ABCMeta

from oolearning.model_processors.SingleUseObject import Cloneable


class HyperParamsBase(Cloneable, metaclass=ABCMeta):
    """
    Each inheriting object is responsible for validating the hyper-parameters' values and ensuring all
    available hyper-parameters are used.

    **Each inheriting object should set `_params_dict` in an overriding constructor.**

    The intent is for the constructor to take the hyperparams as arguments and set the defaults, and possibly
    request additional information necessary to calculate reasonable defaults.
    """
    def __init__(self, match_type=False):
        """
        :param match_type: if True, then in `update_dict()` when updating a value, if the previous value was
            of type int, the new value is first rounded and then converted (used when Optimizers
            e.g. BayesianOptimizationModelTuner passes in float values when integers are required by the model
        """
        self._params_dict = None
        self._match_type = match_type

    @property
    def params_dict(self) -> dict:
        assert self._params_dict is not None  # since each constructor should set this it should never be None
        return self._params_dict

    def update_dict(self, params_dict: dict):
        assert self._params_dict is not None  # since each constructor should set this it should never be None
        # update the object's hyper-parameters dictionary, with the new values from `params_dict`,
        # retaining the default values not specified in `params_dict`
        for key, value in params_dict.items():
            if key not in self._params_dict:
                raise ValueError('key `' + key + '` is not found in current hyper-parameters. Setting a non-existent hyper-parameter is not allowed, because it would not be used anywhere, and is most likely a mistake.')  # noqa

            if self._match_type and isinstance(self._params_dict[key], int):
                value = int(round(value))

            self._params_dict[key] = value
