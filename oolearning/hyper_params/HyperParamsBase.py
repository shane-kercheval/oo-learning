from abc import ABCMeta
import copy


class HyperParamsBase(metaclass=ABCMeta):
    """
    Each inheriting object is responsible for validating the hyper-parameters' values and ensuring all
    available hyper-parameters are used.

    **Each inheriting object should set `_params_dict` in an overriding constructor.**

    The intent is for the constructor to take the hyperparams as arguments and set the defaults, and possibly
    request additional information necessary to calculate reasonable defaults.
    """
    def __init__(self):
        self._params_dict = None

    def clone(self):
        """
        when, for example, tuning, an Resampler will have to be cloned several times (before using)
        :return: a clone of the current object
        """
        return copy.deepcopy(self)

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
            self._params_dict[key] = value
