from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from oolearning.model_processors.SingleUseObject import SingleUseObjectMixin
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelAlreadyFittedError, \
    ModelCachedAlreadyConfigured, ModelNotFittedError
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase


class ModelWrapperBase(SingleUseObjectMixin, metaclass=ABCMeta):
    """
    Intent of ModelWrappers is to abstract away the differences between various types of models and return
    values between models; to give a consistent interface; and rather than passing magic strings and
    configuration parameters, to pass/return objects that have consistent interfaces themselves.
    """

    def __init__(self):

        super().__init__(already_executed_exception_class=ModelAlreadyFittedError,
                         not_executed_exception_class=ModelNotFittedError)
        self._model_object = None
        self._feature_names = None
        self._hyper_params = None
        self._persistence_manager = None
        self._data_x_trained_head = None

    def __str__(self):
        val = self.name

        if self.model_object is not None:
            invert_op = getattr(self.model_object, "get_params", None)
            if callable(invert_op):
                val += "\n\nHyper-Parameters\n================\n\n" + str(self.model_object.get_params()).replace(", ", "\n ")  # noqa

            if self.feature_importance is not None and isinstance(self.feature_importance, pd.DataFrame):
                val += "\n\nFeature Importance\n==================\n\n" + self.feature_importance.round(8).to_string()  # noqa
            else:
                val += "\n\nFeatures Trained\n================\n\n" + str(self._feature_names).replace(", ", "\n ")  # noqa

        return val

    def additional_cloning_checks(self):
        if self._persistence_manager is not None:
            raise ModelCachedAlreadyConfigured('Should not clone after we configure the cache (cloning is intended to reuse empty model objects.).')  # noqa

    def set_persistence_manager(self, persistence_manager: PersistenceManagerBase):
        """
        :param persistence_manager: object that defines show to cache (store/retrieve) the underlying model
        """
        self.ensure_has_not_executed()
        self._persistence_manager = persistence_manager

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def hyper_params(self) -> HyperParamsBase:
        self.ensure_has_executed()
        return self._hyper_params

    @property
    def feature_names(self) -> list:
        self.ensure_has_executed()
        return self._feature_names

    @property
    def data_x_trained_head(self) -> pd.DataFrame:
        """
        :return: the first X rows of the dataset that was trained
        """
        self.ensure_has_executed()
        return self._data_x_trained_head

    @property
    @abstractmethod
    def feature_importance(self):
        pass

    @property
    def model_object(self):
        self.ensure_has_executed()

        return self._model_object

    def train(self,
              data_x: pd.DataFrame,
              data_y: Union[np.ndarray, None] = None,
              hyper_params: HyperParamsBase = None):
        """
        `train()` is friendly name for SingleUseObjectMixin.execute() but both should do the same thing

        trains the model on the training_set; assumes the parent class has transformed/etc. the
        data appropriately

        :param data_x: a 'transformed' DataFrame
        :param data_y: target values
        :param hyper_params: object containing hyper-parameters
        :return: None
        """
        self.execute(data_x=data_x, data_y=data_y, hyper_params=hyper_params)

    def _execute(self,
                 data_x: pd.DataFrame,
                 data_y: Union[np.ndarray, None] = None,
                 hyper_params: HyperParamsBase = None):
        """
        trains the model on the training_set; assumes the parent class has transformed/etc. the
        data appropriately

        :param data_x: a 'transformed' DataFrame
        :param data_y: target values
        :param hyper_params: object containing hyper-parameters
        :return: None
        """

        self._hyper_params = hyper_params
        self._feature_names = data_x.columns.values.tolist()
        self._data_x_trained_head = data_x.head(n=30)

        if self._persistence_manager:
            self._model_object = self._persistence_manager.\
                get_object(fetch_function=lambda: self._train(data_x=data_x,
                                                              data_y=data_y,
                                                              hyper_params=hyper_params))
        else:
            self._model_object = self._train(data_x=data_x, data_y=data_y, hyper_params=hyper_params)

        assert self._model_object is not None

        return None

    def predict(self, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        :type data_x: DataFrame containing the features the model was trained on (without the target column)
        :return: predicted values, actual predictions for Regression models and predicted probabilities for
            Classification problems.
        """
        self.ensure_has_executed()

        # check that the columns passed in via data_x match the columns/features that the model was trained on
        assert len(set(self._feature_names).symmetric_difference(set(data_x.columns.values))) == 0

        predictions = self._predict(model_object=self._model_object, data_x=data_x)
        assert isinstance(predictions, pd.DataFrame) or isinstance(predictions, np.ndarray)
        return predictions

    ##########################################################################################################
    # Abstract Methods
    ##########################################################################################################
    @abstractmethod
    def _train(self,
               data_x: pd.DataFrame,
               data_y: Union[np.ndarray, None],
               hyper_params: HyperParamsBase = None) -> object:
        """
        contains the logic of training the data, to be implemented by the sub-class

        :param data_x:
        :param data_y:
        :param hyper_params:
        :return:
        """
        pass

    @abstractmethod
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        contains the logic to predict the data, to be implemented by the sub-class

        :param model_object:
        :param data_x:
        :return: predicted values, actual predictions for Regression models and predicted probabilities for
            Classification problems.
        """
        raise NotImplementedError()
