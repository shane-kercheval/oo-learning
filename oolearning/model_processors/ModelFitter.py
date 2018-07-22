from typing import List, Callable, Union

import numpy as np
import pandas as pd

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelAlreadyFittedError, ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.transformers.StatelessTransformer import StatelessTransformer
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class ModelFitterRemoving:
    """
    ModelFitter encapsulates the (mundane and repetitive) logic of the general process of "training" an
        unsupervised model i.e. we aren't "training" a model against a target variable (that is, no `data_y`
        in `train_predict_eval()`.

        including:

        - data transformations & pre-processing
        - training i.e. fitting a model

    Very similar to ModelTrainer, except it does not have various unused functionality (e.g. splitting data)
    """

    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase]=None,
                 persistence_manager: PersistenceManagerBase=None,
                 fit_callback: Callable[[pd.DataFrame,
                                         Union[HyperParamsBase, None]], None] = None):
        """

        :param model: a class representing the model to train_predict_eval
        :param model_transformations: a list of transformations to apply before training (and predicting)
        :param persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
        :param fit_callback: a callback that is called before the model is trained, which returns the
           data_x, data_y, and hyper_params that are passed into `ModelWrapper.train_predict_eval()`.
           The primary intent is for unit tests to have the ability to ensure that the data (data_x) is
           being transformed as expected, but it is imaginable to think that users will also benefit
           from this capability to also peak at the data that is being trained.
        """
        assert isinstance(model, ModelWrapperBase)
        self._model = model
        # copy so that we can use 'same' evaluator type in the holdout evaluator
        self._has_fitted = False
        self._persistence_manager = persistence_manager
        self._fit_callback = fit_callback

        if model_transformations is not None:
            assert isinstance(model_transformations, list)
            assert all([isinstance(x, TransformerBase) for x in model_transformations])

        self._model_transformations = model_transformations
        self._pipeline = None

    @property
    def model(self) -> ModelWrapperBase:
        """
        :return: underlying model object
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._model

    def set_persistence_manager(self, persistence_manager: PersistenceManagerBase):
        """
        Sets the persistence manager, defining how the underlying model should be cached
        :param persistence_manager:
        :return:
        """
        self._persistence_manager = persistence_manager

    @staticmethod
    def _build_cache_key(model: ModelWrapperBase, hyper_params: HyperParamsBase) -> str:
        """
        helper function to build the cache key (e.g. file name)
        """
        model_name = model.name
        if hyper_params is None:
            key = model_name
        else:
            # if hyper-params, flatten out list of param names and values and concatenate/join them together
            hyper_params_long = '_'.join(list(sum([(str(x), str(y)) for x, y in hyper_params.params_dict.items()], ())))  # noqa
            return model_name + '_' + hyper_params_long

        return key

    def fit(self, data: pd.DataFrame, hyper_params: HyperParamsBase=None):
        if self._has_fitted:
            raise ModelAlreadyFittedError()

        # transform/train_predict_eval on training data
        if self._model_transformations is not None:
            # before we train_predict_eval the data, we actually want to 'snoop' at what the expected columns will be with
            # ALL the data. The reason is that if we so some sort of dummy encoding, but not all the
            # categories are included in the training set (i.e. maybe only a small number of observations have
            # the categoric value), then we can still ensure that we will be giving the same expected columns/
            # encodings to the predict method with the holdout set.
            # noinspection PyTypeChecker
            expected_columns = TransformerPipeline.get_expected_columns(data=data,
                                                                        transformations=self._model_transformations)  # noqa
            transformer = StatelessTransformer(custom_function=lambda x_df: x_df.reindex(columns=expected_columns,  # noqa
                                                                                         fill_value=0))
            self._model_transformations = self._model_transformations + [transformer]

        self._pipeline = TransformerPipeline(transformations=self._model_transformations)
        # before we fit the data, we actually want to 'peak' at what the expected columns will be with
        # ALL the data. The reason is that if we so some sort of encoding (dummy/one-hot), but not all
        # of the categories are included in the training set (i.e. maybe only a small number of
        # observations have the categoric value), then we can still ensure that we will be giving the
        # same expected columns/encodings to the `predict` method with the holdout set.

        # peak at all the data (except for the target variable of course)
        # noinspection PyTypeChecker
        self._pipeline.peak(data_x=data)
        # fit on only the train_predict_eval data-set (and also transform)
        transformed_data = self._pipeline.fit_transform(data_x=data)

        # set up persistence if applicable
        if self._persistence_manager is not None:  # then build the key
            cache_key = ModelFitterRemoving._build_cache_key(model=self._model, hyper_params=hyper_params)
            self._persistence_manager.set_key(key=cache_key)
            self._model.set_persistence_manager(persistence_manager=self._persistence_manager)

        if self._fit_callback is not None:
            self._fit_callback(transformed_data, hyper_params)

        # train_predict_eval the model with the transformed training data
        self._model.train(data_x=transformed_data, hyper_params=hyper_params)

        self._has_fitted = True

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        `predict` handles the logic of applying the transformations (same transformations that were applied to
            the training data, as well as predicted data
        :param data: unprocessed DataFrame (unprocessed in terms of the model specific transformation
            pipeline, i.e. exactly the same transformations should be applied to this data as was used on the
            training data
        :return: predicted values
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        prepared_prediction_set = self._pipeline.transform(data)

        predictions = self._model.predict(data_x=prepared_prediction_set)

        return predictions

    def fit_predict(self, data, hyper_params):
        self.fit(data=data, hyper_params=hyper_params)
        return self.predict(data=data)
