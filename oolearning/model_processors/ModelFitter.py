import copy
from typing import List, Callable, Union

import numpy as np
import pandas as pd

from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.fitted_info.FittedInfoBase import FittedInfoBase
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelAlreadyFittedError, ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.splitters.DataSplitterBase import DataSplitterBase
from oolearning.transformers.StatelessTransformer import StatelessTransformer
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class ModelFitter:
    """
    # TODO document
    Intent of ModelFitter is to abstract away the details of the general process of fitting a model.
        - transform data specific to the model (e.g. regression requires imputing/dummifying
        - train
        - access training value
        - evaluate on a holdout set
        - predict on future data using same (training) transformations
    """

    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase],
                 splitter: DataSplitterBase,
                 evaluator: Union[EvaluatorBase]=None,
                 scores: Union[List[ScoreBase]]=None,
                 persistence_manager: PersistenceManagerBase=None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None):
        """
        # TODO: update documentation
        :param model_transformations: List of Transformer objects to pre-process data (specific to the model,
        e.g. Regression should impute and create dummy columns).
            Child classes should list recommended transformations as the default value to the constructor
            and callers have the ability to replace with their own list (or None)
        :param persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
        :param train_callback: a callback that is called before the model is trained, which returns the
            data_x, data_y, and hyper_params that are passed into `ModelWrapper.train()`.
            The primary intent is for unit tests to have the ability to ensure that the data (data_x) is
            being transformed as expected, but it is imaginable to think that users will also benefit
            from this capability to also peak at the data that is being trained.
        """
        assert isinstance(model, ModelWrapperBase)
        self._model = model
        self._splitter = splitter
        self._training_evaluator = evaluator
        # copy so that we can use 'same' evaluator type
        self._holdout_evaluator = copy.deepcopy(evaluator)
        self._training_scores = scores
        self._holdout_scores = None if scores is None else [x.clone() for x in scores]
        self._has_fitted = False
        self._model_info = None
        self._persistence_manager = persistence_manager
        self._train_callback = train_callback

        if model_transformations is not None:
            assert isinstance(model_transformations, list)
            assert all([isinstance(x, TransformerBase) for x in model_transformations])

        self._model_transformations = model_transformations
        self._pipeline = None

    @property
    def model_info(self) -> FittedInfoBase:
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._model_info

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

    def fit(self, data: pd.DataFrame, target_variable: str, hyper_params: HyperParamsBase=None):
        if self._has_fitted:
            raise ModelAlreadyFittedError()

        training_indexes, holdout_indexes = self._splitter.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns=target_variable)

        holdout_y = data.iloc[holdout_indexes][target_variable]
        holdout_x = data.iloc[holdout_indexes].drop(columns=target_variable)

        # transform/fit on training data
        if self._model_transformations is not None:
            # before we fit the data, we actually want to 'snoop' at what the expected columns will be with
            # ALL the data. The reason is that if we so some sort of dummy encoding, but not all the
            # categories are included in the training set (i.e. maybe only a small number of observations have
            # the categoric value), then we can still ensure that we will be giving the same expected columns/
            # encodings to the predict method with the holdout set.
            # noinspection PyTypeChecker
            expected_columns = TransformerPipeline.get_expected_columns(data=data.drop(columns=target_variable),  # noqa
                                                                        transformations=self._model_transformations)  # noqa
            transformer = StatelessTransformer(custom_function=
                                               lambda x_df: x_df.reindex(columns=expected_columns,
                                                                         fill_value=0))
            self._model_transformations = self._model_transformations + [transformer]

        self._pipeline = TransformerPipeline(transformations=self._model_transformations)
        transformed_training_data = self._pipeline.fit_transform(training_x)

        # set up persistence if applicable
        if self._persistence_manager is not None:  # then build the key
            cache_key = ModelFitter.build_cache_key(model=self._model, hyper_params=hyper_params)
            self._persistence_manager.set_key(key=cache_key)
            self._model.set_persistence_manager(persistence_manager=self._persistence_manager)

        if self._train_callback is not None:
            self._train_callback(transformed_training_data, training_y, hyper_params)

        # fit the model with the transformed training data
        self._model.train(data_x=transformed_training_data, data_y=training_y, hyper_params=hyper_params)
        self._model_info = self._model.fitted_info

        self._has_fitted = True

        # if evaluators, evaluate on both the training and holdout set
        if self._training_evaluator is not None:
            # predict will apply the transformations (which are fitted on the training data)
            self._training_evaluator.evaluate(actual_values=training_y,
                                              predicted_values=self.predict(data_x=training_x))
            self._holdout_evaluator.evaluate(actual_values=holdout_y,
                                             predicted_values=self.predict(data_x=holdout_x))

        # if scores, score on both the training and holdout set
        if self._training_scores is not None:
            # predict will apply the transformations (which are fitted on the training data)
            for score in self._training_scores:
                score.calculate(actual_values=training_y,
                                predicted_values=self.predict(data_x=training_x))
            for score in self._holdout_scores:
                score.calculate(actual_values=holdout_y,
                                predicted_values=self.predict(data_x=holdout_x))

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

        prepared_prediction_set = self._pipeline.transform(data_x)
        return self._model.predict(data_x=prepared_prediction_set)

    @property
    def training_evaluator(self):
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._training_evaluator

    @property
    def holdout_evaluator(self):
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._holdout_evaluator

    @property
    def training_scores(self):
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._training_scores

    @property
    def holdout_scores(self):
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._holdout_scores
