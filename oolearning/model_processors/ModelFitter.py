import copy
from typing import List

import numpy as np
import pandas as pd

from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.fitted_info.FittedInfoBase import FittedInfoBase
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelAlreadyFittedError, ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class ModelFitter:
    """
    Intent of ModelFitter is to abstract away the details of the general process of fitting a model.
        - transform data specific to the model (e.g. regression requires imputing/dummifying
        - train
        - access training value
        - access holdout value
        - predict on future data using same (training) transformations
    """

    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase],
                 evaluators: List[EvaluatorBase],
                 persistence_manager: PersistenceManagerBase=None):
        """
        :param evaluators: a list of Evaluator to access training value.
            If no evaluator is passed into the `evaluate_holdout()` method, this evaluator is cloned and also
            used to evaluate a holdout dataset
        :param model_transformations: List of Transformer objects to pre-process data (specific to the model,
        e.g. Regression should impute and create dummy columns).
            Child classes should list recommended transformations as the default value to the constructor
            and callers have the ability to replace with their own list (or None)
        :param persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
        """
        assert isinstance(model, ModelWrapperBase)
        self._model = model
        self._training_evaluators = evaluators
        # copy so that we can use 'same' evaluator types
        self._holdout_evaluators = [x.clone() for x in copy.deepcopy(evaluators)]
        self._has_fitted = False
        self._model_info = None
        self._persistence_manager = persistence_manager

        if model_transformations is not None:
            assert isinstance(model_transformations, list)
            assert all([isinstance(x, TransformerBase) for x in model_transformations])

        self._model_transformations = TransformerPipeline(transformations=model_transformations)

    @property
    def model_info(self) -> FittedInfoBase:
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._model_info

    @property
    def training_evaluators(self) -> List[EvaluatorBase]:
        """
        :return: the 'Training Evaluators' passed into the constructor
        """
        return self._training_evaluators

    @property
    def training_accuracies(self) -> List[float]:
        """
        :return: the "accuracies" returned by the 'training_evaluators'
        """
        assert all([isinstance(x.value, float) for x in self._training_evaluators])
        return [x.value for x in self._training_evaluators]

    @property
    def holdout_evaluators(self) -> List[EvaluatorBase]:
        """
        :return: the 'Training Evaluators' passed into the constructor
        """
        return self._holdout_evaluators

    @property
    def holdout_accuracies(self) -> List[float]:
        """
        :return: returns the "accuracies" returned by the 'holdout_evaluators'
        """
        assert all([isinstance(x.value, float) for x in self._holdout_evaluators])
        return [x.value for x in self._holdout_evaluators]

    def set_persistence_manager(self, persistence_manager: PersistenceManagerBase):
        self._persistence_manager = persistence_manager

    @staticmethod
    def build_cache_key(model: ModelWrapperBase, hyper_params: HyperParamsBase) -> str:
        model_name = type(model).__name__
        if hyper_params is None:
            key = model_name
        else:
            # if hyper-params, flatten out list of param names and values and concatenate/join them together
            hyper_params_long = '_'.join(list(sum([(str(x), str(y)) for x, y in hyper_params.params_dict.items()], ())))  # noqa
            return model_name + '_' + hyper_params_long

        return key

    def fit(self,
            data_x: pd.DataFrame,
            data_y: np.ndarray,
            hyper_params: HyperParamsBase=None):
        """
        `fit` handles the logic of applying the pre-process transformations, as well as fitting the data and
            evaluating the training value
        :param data_x: DataFrame to fit the model on
        :param data_y: np.ndarray containing the target values to be trained on
        :param hyper_params: object containing the hyper-parameters to tune
        :return: None
        """
        if self._has_fitted:
            raise ModelAlreadyFittedError()

        assert isinstance(data_x, pd.DataFrame)
        assert data_y is not None

        prepared_training_data = self._model_transformations.fit_transform(data_x)

        # set up persistence if applicable
        if self._persistence_manager is not None:  # then build the key
            cache_key = ModelFitter.build_cache_key(model=self._model, hyper_params=hyper_params)
            self._persistence_manager.set_key(key=cache_key)
            self._model.set_persistence_manager(persistence_manager=self._persistence_manager)

        # fit the model with the transformed training data
        self._model.train(data_x=prepared_training_data,
                          data_y=data_y,
                          hyper_params=hyper_params)

        self._model_info = self._model.fitted_info

        for evaluator in self._training_evaluators:
            # given the specified **training** metric, which stores the value
            evaluator.evaluate(actual_values=data_y,
                               predicted_values=self._model.predict(data_x=prepared_training_data))
        self._has_fitted = True

    def predict(self, data_x: pd.DataFrame) -> np.ndarray:
        """
        `predict` handles the logic of applying the transformations (same transformations that were applied to
            the training data, as well as predicted data
        :param data_x: unprocessed DataFrame (unprocessed in terms of the model specific transformation
            pipeline, i.e. exactly the same transformations should be applied to this data as was used on the
            training data
        :return: predicted values
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        prepared_prediction_set = self._model_transformations.transform(data_x)
        return self._model.predict(data_x=prepared_prediction_set)

    def evaluate_holdout(self,
                         holdout_x: pd.DataFrame,
                         holdout_y: np.ndarray,
                         evaluators: List[EvaluatorBase]=None) -> List[float]:
        """
        :param holdout_x: holdout dataset with features
        :param holdout_y: holdout target values
        :param evaluators: optional list of Evaluator. If `holdout_evaluators` is None, a clone of the
            holdout_evaluators that were passed into the constructor is used (i.e. same holdout_evaluators
            type)
        :return: The result of the Evaluators` `evaluate()` function (i.e. "value")
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        if evaluators is not None:  # otherwise, use same evaluator as training set (previously cloned)
            self._holdout_evaluators = evaluators
        # the evaluator was cloned before used, so if there is no evaluator passed in, we'll use that
        accuracies = list()
        for evaluator in self._holdout_evaluators:
            accuracy = evaluator.evaluate(actual_values=holdout_y,
                                          predicted_values=self.predict(data_x=holdout_x))
            accuracies.append(accuracy)

        return accuracies
