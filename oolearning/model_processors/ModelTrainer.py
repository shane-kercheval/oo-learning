import copy
from typing import List, Callable, Union

import numpy as np
import pandas as pd

from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.evaluators.ScoreMediator import ScoreMediator
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelAlreadyFittedError, ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.splitters.DataSplitterBase import DataSplitterBase
from oolearning.transformers.StatelessTransformer import StatelessTransformer
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class ModelTrainer:
    """
    ModelTrainer encapsulates the (mundane and repetitive) logic of the general process of training a model,
        including:

        - splitting the data into training and holdout sets
        - data transformations & pre-processing
        - training a model
        - predicting on a holdout data-set, or on future data (applying the same transformations)
        - evaluate the performance of the model on a holdout set
    """

    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: Union[List[TransformerBase], None]=None,
                 splitter: DataSplitterBase=None,
                 evaluator: EvaluatorBase=None,
                 scores: List[ScoreBase]=None,
                 persistence_manager: PersistenceManagerBase=None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None):
        """

        :param model: a class representing the model to train_predict_eval
        :param model_transformations: a list of transformations to apply before training (and predicting)
        :param splitter: a class encapsulating the logic of splitting the data into training and holdout sets;
            if None, then no split occurs, and the model is trained on all the data (and so no holdout
            evaluator or scores are available).
        :param evaluator: a class encapsulating the logic of evaluating a holdout set
        :param scores: a list of Score objects
        :param persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
        :param train_callback: a callback that is called before the model is trained, which returns the
           data_x, data_y, and hyper_params that are passed into `ModelWrapper.train_predict_eval()`.
           The primary intent is for unit tests to have the ability to ensure that the data (data_x) is
           being transformed as expected, but it is imaginable to think that users will also benefit
           from this capability to also peak at the data that is being trained.
        """
        assert isinstance(model, ModelWrapperBase)
        self._model = model
        self._splitter = splitter
        self._training_evaluator = evaluator
        # copy so that we can use 'same' evaluator type in the holdout evaluator
        self._holdout_evaluator = copy.deepcopy(evaluator)
        self._training_scores = scores
        self._holdout_scores = None if scores is None else [x.clone() for x in scores]
        self._has_fitted = False
        self._persistence_manager = persistence_manager
        self._train_callback = train_callback

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

    def train_predict_eval(self,
                           data: pd.DataFrame,
                           target_variable: Union[str, None]=None,
                           hyper_params: HyperParamsBase=None) -> np.ndarray:
        """
        The data is split into a training/holdout set if a Splitter is provided. If not provided, no split
            occurs and the model is trained on all the `data`). Before training, the data is transformed
            by the specified Transformation objects. If a Splitter is provided, the transformations are
            'fit/transformed' on the training and only transformed on the holdout.

        Trains the data on the model, predicts, and evaluates the predictions if an Evaluator or Scores are
            passed in.
            If a Splitter is provide, the predictions that are returned are of the holdout set. Otherwise,
            the predictions form the training set are returned.

        :param data: data to split (if Splitter is provided) and train_predict_eval the model on
        :param target_variable: the name of the target variable/column
        :param hyper_params: a corresponding HyperParams object
        """
        if self._has_fitted:
            raise ModelAlreadyFittedError()

        if self._splitter:
            assert target_variable is not None
            training_indexes, holdout_indexes = self._splitter.split(target_values=data[target_variable])
        else:  # we are fitting the entire data-set, no such thing as a holdout data-set/evaluator/scores
            training_indexes, holdout_indexes = range(len(data)), []
            self._holdout_evaluator = None
            self._holdout_scores = None

        # for unsupervised problems, there might not be a target variable;
        # in that case, there will also not be a training_y/holding_y
        training_y = data.iloc[training_indexes][target_variable] if target_variable is not None else None
        training_x = data.iloc[training_indexes]

        holdout_y = data.iloc[holdout_indexes][target_variable] if target_variable is not None else None
        holdout_x = data.iloc[holdout_indexes]

        if target_variable is not None:
            training_x = training_x.drop(columns=target_variable)
            holdout_x = holdout_x.drop(columns=target_variable)

        # transform/train_predict_eval on training data
        if self._model_transformations is not None:
            # before we train_predict_eval the data, we actually want to 'snoop' at what the expected columns
            # will be with ALL the data. The reason is that if we so some sort of dummy encoding, but not all
            # the categories are included in the training set (i.e. maybe only a small number of observations
            # have the categoric value), then we can still ensure that we will be giving the same expected
            # columns/encodings to the predict method with the holdout set.
            expected_columns = TransformerPipeline.\
                get_expected_columns(data=data if target_variable is None else data.drop(columns=target_variable),  # noqa
                                     transformations=self._model_transformations)
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
        self._pipeline.peak(data_x=data if target_variable is None else data.drop(columns=target_variable))
        # fit on only the train_predict_eval data-set (and also transform)
        transformed_training_data = self._pipeline.fit_transform(training_x)

        # set up persistence if applicable
        if self._persistence_manager is not None:  # then build the key
            cache_key = ModelTrainer._build_cache_key(model=self._model, hyper_params=hyper_params)
            self._persistence_manager.set_key(key=cache_key)
            self._model.set_persistence_manager(persistence_manager=self._persistence_manager)

        if self._train_callback is not None:
            self._train_callback(transformed_training_data, training_y, hyper_params)

        # train_predict_eval the model with the transformed training data
        self._model.train(data_x=transformed_training_data, data_y=training_y, hyper_params=hyper_params)

        self._has_fitted = True

        training_predictions = self.predict(data_x=training_x)
        holdout_predictions = None
        if self._splitter is not None:
            holdout_predictions = self.predict(data_x=holdout_x)

        # if evaluators, evaluate on both the training and holdout set
        if self._training_evaluator is not None:
            # predict will apply the transformations (which are fitted on the training data)
            self._training_evaluator.evaluate(actual_values=training_y,
                                              predicted_values=training_predictions)
            if self._holdout_evaluator:
                self._holdout_evaluator.evaluate(actual_values=holdout_y,
                                                 predicted_values=holdout_predictions)

        # if scores, score on both the training and holdout set
        if self._training_scores is not None:
            # predict will apply the transformations (which are fitted on the training data)
            for score in self._training_scores:
                ScoreMediator.calculate(score=score,
                                        data_x=transformed_training_data,
                                        actual_target_variables=training_y,
                                        predicted_values=training_predictions)

            if self._holdout_scores:
                for score in self._holdout_scores:
                    ScoreMediator.calculate(score=score,
                                            data_x=holdout_x,  # TODO may have to manually do transformations
                                            actual_target_variables=holdout_y,
                                            predicted_values=holdout_predictions)

        return training_predictions if self._splitter is None else holdout_predictions

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

        predictions = self._model.predict(data_x=prepared_prediction_set)
        if isinstance(predictions, pd.DataFrame):
            # noinspection PyTypeChecker
            assert all(predictions.index.values == data_x.index.values)

        return predictions

    @property
    def training_evaluator(self) -> Union[EvaluatorBase, None]:
        """
        :return: if an Evaluator was provided via class constructor, returns the object evaluated on the
            training data
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._training_evaluator

    @property
    def holdout_evaluator(self) -> Union[EvaluatorBase, None]:
        """
        :return: if an Evaluator *and* a Splitter (thus creating a holdout set before training) were provided
            via class constructor, returns the object evaluated on the holdout data
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._holdout_evaluator

    @property
    def training_scores(self) -> Union[List[ScoreBase], None]:
        """
        :return: if a list of Scores was provided via class constructor, returns the list of Scores calculated
            on the training data.
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._training_scores

    @property
    def holdout_scores(self) -> Union[List[ScoreBase], None]:
        """
        :return: if list of Scores *and* a Splitter (thus creating a holdout set before training) were
            provided via class constructor, returns the list of Scores evaluated on the holdout data
        """
        if self._has_fitted is False:
            raise ModelNotFittedError()

        return self._holdout_scores
