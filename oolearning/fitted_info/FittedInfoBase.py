from abc import ABCMeta, abstractmethod
from typing import List

from matplotlib import figure

from oolearning.hyper_params.HyperParamsBase import HyperParamsBase


class FittedInfoBase(metaclass=ABCMeta):
    """
    Class for wrapping the information about a fitted model
    """

    def __init__(self,
                 model_object: object,
                 feature_names: List[str],
                 hyper_params: HyperParamsBase=None):
        self._model_object = model_object
        self._feature_names = feature_names
        self._hyper_params = hyper_params

    @property
    @abstractmethod
    def results_summary(self) -> object:
        """
        :return: for sub-class to define, e.g. Regression could return feature coefficients and p-values
        """
        # noinspection PyUnresolvedReferences
        return self.model_object.summary()

    @property
    @abstractmethod
    def summary_stats(self) -> dict:
        """
        :return: for sub-class to define, e.g. Regression could return RSE, adjusted R^2, etc.
        """
        pass

    @property
    @abstractmethod
    def warnings(self):
        """
        :return: for sub-class to define, e.g. any violated assumptions of the model
        """
        pass    

    @property
    @abstractmethod
    def feature_importance(self) -> dict:
        """
        :return: ranking of importance of features the model was trained with
        """
        pass

    @property
    @abstractmethod
    def graph(self) -> figure.Figure:
        """
        :return: for sub-class to define, e.g. Regression could return graph of residuals vs fits, etc.
        """
        pass

    @property
    def model_object(self):
        """
        :return: the object that was returned by the underlying model, which is implementation specific
        """
        return self._model_object

    @property
    def feature_names(self) -> List[str]:
        """
        :return: a list of the feature names that were fitted by the model
        """
        return self._feature_names

    @property
    def hyper_params(self) -> HyperParamsBase:
        """
        :return: a list of the hyper-parameters that were used to fit the model
        """
        return self._hyper_params
