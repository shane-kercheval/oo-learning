import copy
import numpy as np
import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import Union

from oolearning.persistence.AlwaysFetchManager import AlwaysFetchManager
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.fitted_info.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError, ModelAlreadyFittedError,\
    ModelCachedAlreadyConfigured


class ModelWrapperBase(metaclass=ABCMeta):
    """
    Intent of ModelWrappers is to abstract away the differences between various types of models and return
    values between models; to give a consistent interface; and rather than passing magic strings and
    configuration parameters, to pass/return objects that have consistent interfaces themselves.
    """

    def __init__(self):
        self._fitted_info = None
        # set up the PersistenceManager, default it to NoCacheManager which just returns the object from the
        # function passed in (i.e. it isn't stored anywhere, we have to go get it or create it)
        # don't allow it to be configured in constructor in case object is cloned
        self._persistence_manager = AlwaysFetchManager()

    @property
    def fitted_info(self) -> FittedInfoBase:
        """
        :return: returns information about the model after it is fitted
        """
        if self._fitted_info is None:
            raise ModelNotFittedError()

        return self._fitted_info

    def clone(self):
        """
        when, for example, resampling, a model will have to be cloned several times (before fitting)
        :return: a clone of the current object
        """
        if self._fitted_info is not None:  # only intended on being called before fitting
            raise ModelAlreadyFittedError()

        # if the PersistenceManager is not the default object (i.e. no cache)
        if not isinstance(self._persistence_manager, AlwaysFetchManager):
            raise ModelCachedAlreadyConfigured('Should not clone after we configure the cache (cloning is intended to reuse empty model objects.).')  # noqa

        return copy.deepcopy(self)

    def set_persistence_manager(self, persistence_manager):
        """
        #TODO: fix documentation
        # NOTE: cannot pass cache_path in to constructor, in case we want to clone the model.
        :param persistence_manager:
        :param persistence_manager: cache (store/retrieve) the underlying model
        #TODO: document: so, the workflow is the same whether or not you are retrieving an existing cache or not... i.e. you cannot go from retreiving to predicting without "training", even if the model is cached, before the train function passes important information to the FittedInfo object
        """
        if self._fitted_info is not None:  # doesn't make sense to configure the cache after we `train()`
            raise ModelAlreadyFittedError()

        self._persistence_manager = persistence_manager

    def train(self,
              data_x: pd.DataFrame,
              data_y: np.ndarray,
              hyper_params: HyperParamsBase=None):
        """
        trains the model on the training_set; assumes the parent class has transformed/etc. the
        data appropriately

        :param data_x: a 'transformed' DataFrame
        :param data_y: target values
        :param hyper_params: object containing hyper-parameters
        :return: None
        """

        # if _fitted_info is not None, then we have already trained/fitted
        if self._fitted_info is not None:
            raise ModelAlreadyFittedError()

        # this gets the object based on the persistence system of _persistence_object;
        # the default object type, AlwaysFetchObject, always creates the object i.e. always calls _train
        model_object = self._persistence_manager.get_object(
            fetch_function=lambda: self._train(data_x=data_x, data_y=data_y, hyper_params=hyper_params))
        assert model_object is not None

        # now that we have the underlying model, allow the base class to save whatever it needs to save
        self._fitted_info = self._create_fitted_info_object(model_object=model_object,
                                                            data_x=data_x,
                                                            data_y=data_y,
                                                            hyper_params=hyper_params)

    def predict(self, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        :type data_x: DataFrame containing the features the model was trained on (without the target column)
        :return: predicted values, actual predictions for Regression models and predicted probabilities for
            Classification problems.
        """
        if self._fitted_info is None:
            raise ModelNotFittedError()

        # check that the columns passed in via data_x match the columns/features that the model was trained on
        assert len(set(self._fitted_info.feature_names).symmetric_difference(set(data_x.columns.values))) == 0

        return self._predict(model_object=self._fitted_info.model_object, data_x=data_x)

    ##########################################################################################################
    # Abstract Methods
    ##########################################################################################################
    @abstractmethod
    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase=None) -> object:
        """
        contains the logic of training the data, to be implemented by the sub-class

        :param data_x:
        :param data_y:
        :param hyper_params:
        :return:
        """
        pass

    @abstractmethod
    def _create_fitted_info_object(self,
                                   model_object,
                                   data_x: pd.DataFrame,
                                   data_y: np.ndarray,
                                   hyper_params: HyperParamsBase=None) -> FittedInfoBase:
        #TODO: add documentation, explain we we need this (i.e. so base classes can save any info they need specifically, while allowing the base class to cache the model_object
        pass

    @abstractmethod
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        contains the logic to predict the data, to be implemented by the sub-class

        :param model_object: the object stored in FittedInfoBase.ModelObject in `_train()`
        :param data_x:
        :return: predicted values, actual predictions for Regression models and predicted probabilities for
            Classification problems.
        """
        pass
