from typing import Union, List, Callable
from multiprocessing import Lock

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oolearning.OOLearningHelpers import OOLearningHelpers, Singleton
from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_processors.RepeatedCrossValidationResampler import RepeatedCrossValidationResampler
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError, ModelAlreadyFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class FoldPredictionsDecorator(DecoratorBase):
    """
    decorator that is passed into the Resampler and, for each fold of the resampler, extracts the holdout's
    predicted values and corresponding indices of the holdout (of the original data-set (i.e. training set)
    passed in). The predictions and corresponding indices are used to build up `train_meta` in the model
    stacker. `train_meta` contains the predictions from each base model (of the training set), and is used
    to train the stacker.
    """
    def __init__(self):
        self._holdout_indexes = list()
        self._holdout_predicted_values = None

    def decorate(self, **kwargs):
        self._holdout_indexes.extend(kwargs['holdout_indexes'])
        predicted_values = kwargs['holdout_predicted_values']
        # predicted_values is either a DataFrame in the case of classification ModelWrappers,
        # or an array in the case of regression ModelWrappers
        if isinstance(predicted_values, pd.DataFrame):
            if self._holdout_predicted_values is None:
                self._holdout_predicted_values = pd.DataFrame()

            self._holdout_predicted_values = self._holdout_predicted_values.append(predicted_values)  # noqa
        else:
            if self._holdout_predicted_values is None:
                self._holdout_predicted_values = np.array([])

            self._holdout_predicted_values = np.append(self._holdout_predicted_values, predicted_values)

    @property
    def holdout_indexes(self):
        return self._holdout_indexes

    @property
    def holdout_predicted_values(self):
        return self._holdout_predicted_values


class ModelStackerTrainingObject:
    """
    In a typical model wrapper, the model_object is returned from `_train` and used in `_predict` with the
    assumption of independence; i.e. _predict doesn't use anything else than the model_object it is given
    when the persistence_manager finds a record, it doesn't even call train, it just returns the model_object
    it de-serialized. So we have to use `_predict` as if it didn't have access to class variables set up in
    `_train`, in order to use the persistence manager
    """
    def __init__(self, base_models: List[ModelInfo],
                 base_model_pipelines: List[TransformerPipeline],
                 stacking_model: ModelWrapperBase,
                 stacking_model_pipeline: Union[TransformerPipeline, None]):
        self._base_models = base_models
        self._base_model_pipelines = base_model_pipelines
        self._stacking_model = stacking_model
        self._stacking_model_pipeline = stacking_model_pipeline

    @property
    def base_models(self) -> List[ModelInfo]:
        return self._base_models

    @property
    def base_model_pipelines(self) -> List[TransformerPipeline]:
        return self._base_model_pipelines

    @property
    def stacking_model(self) -> ModelWrapperBase:
        return self._stacking_model

    @property
    def stacking_model_pipeline(self) -> Union[TransformerPipeline, None]:
        return self._stacking_model_pipeline


class ModelStacker(ModelWrapperBase):
    """
    NOTE: do not use caching/persistence-manager when tuning with ModelStacker, there is an unknown
        race condition when working with the cached files. It needs to be investigated.
        https://github.com/shane-kercheval/oo-learning/issues/44

    This class implements a "Model Stacker" adopted from
    http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/

    The assumption in flow is that each specific model (base models & stacker) will have been previously
        cross-validated in order to choose the best hyper-params for the specific model. As the blog post
        mentions, there might be leakage from this method. However, if predicted on a holdout set never
        seen by the actual models trained, it should still be a fair representation of the test error. 

    The steps for building the stacker are as follows (again, directly taken from the blog, above:

    1. Partition the training data into five test folds (note: 5 could be refactored as parameter)
    2. Create a data-set called train_meta with the same row Ids and fold Ids as the training data-set, with
        empty columns M1 and M2.
        Similarly create a data-set called test_meta with the same row Ids as the test data-set and empty
            columns M1 and M2 (NOTE: this will be in the `_predict` function

    3. For each test fold
        3.1 Combine the other four folds to be used as a training fold
        3.2 For each base model (with chosen hyper-params)
        3.2.1 Fit the base model to the training fold and make predictions on the test fold.
            Store these predictions in train_meta to be used as features for the stacking model
            NOTE: i will also have to do the model specific Transformations

    4. Fit each base model to the full training data-set and make predictions on the test data-set.
        Store these predictions inside test_meta
        NOTE: i will make predictions as part of the `_predict` function

    5. Fit a new model, S (i.e the stacking model) to train_meta, using M1 and M2 as features.
        Optionally, include other features from the original training data-set or engineered features.

    6. Use the stacked model S to make final predictions on test_meta
        NOTE: this will be in `_predict`
    """
    def __init__(self,
                 base_models: List[ModelInfo],
                 scores: List[ScoreActualPredictedBase],
                 stacking_model: ModelWrapperBase,
                 stacking_transformations: List[TransformerBase]=None,
                 include_original_dataset: bool=False,
                 converter: ContinuousToClassConverterBase=None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None,
                 predict_callback: Callable[[pd.DataFrame], None] = None):
        """
        :param base_models:
        :param stacking_transformations: The transformations that are specific to the 'stacker'.
            (The transformations that are specific to the base models will be included in the corresponding
            ModelInfo objects. Transformations applied to ALL models (base and stacker) could be passed in
            via (for example) a ModelTrainer.)
        :param include_original_dataset: When training the 'stacker' (using the predictions of the base models
            as features), this parameter indicates whether or not the original dataset should be also included
            as features, along with the base-model predictions.
        :param scores: since we are cross-validating, we can get a score from each base-model
        :param converter: A Converter object specifying how the predictions (e.g. DataFrame of probabilities
            for a classification problem) should be converted to classes. If the base_models `predict()`
            returns a DataFrame, then a `converter` will need to be supplied to describe how to extract
            the predictions for the positive class,
                e.g. `converter=ExtractPredictionsColumnConverter(column=positive_class)`
        """
        super().__init__()
        # ensure unique model descriptions
        assert len(set([x.description for x in base_models])) == len(base_models)

        self._base_models = base_models
        self._scores = scores
        self._stacking_model = stacking_model
        self._stacking_model_pipeline = None if stacking_transformations is None \
            else TransformerPipeline(transformations=stacking_transformations)
        self._include_original_dataset = include_original_dataset
        self._converter = converter
        self._resampler_results = list()
        self._base_model_pipelines = list()
        self._train_callback = train_callback
        self._predict_callback = predict_callback
        self._train_meta_correlations = None
        self._stackerobject_persistence_manager = None

    class StackerMetaLock(metaclass=Singleton):
        """
        Example class.
        """

        def __init__(self):
            self._lock = Lock()

        def acquire(self):
            self._lock.acquire()

        def release(self):
            self._lock.release()

    class StackerBaseLock(metaclass=Singleton):
        """
        Example class.
        """

        def __init__(self):
            self._lock = Lock()

        def acquire(self):
            self._lock.acquire()

        def release(self):
            self._lock.release()

    @property
    def name(self) -> str:
        return "{0}_{1}".format(type(self).__name__, type(self._stacking_model).__name__)

    def get_resample_data(self, model_description: str) -> pd.DataFrame:
        """
        :param model_description: the description of the model (i.e. identifier passed into the ModelInfo
            object)
        :return: a DataFrame containing the resampled scores for each holdout for a given model.
        """
        if self._model_object is None:
            raise ModelNotFittedError()

        model_index = [x.description for x in self._base_models].index(model_description)
        return self._resampler_results[model_index].resampled_scores

    def get_resample_means(self) -> pd.DataFrame:
        """
        :return: a DataFrame containing the Scores (as rows) for each model (as columns)
        """
        if self._model_object is None:
            raise ModelNotFittedError()

        score_names = [x.name for x in self._scores]
        model_names = [x.description for x in self._base_models]
        resample_means = pd.DataFrame(index=score_names,
                                      columns=model_names)
        for model in model_names:
            resample_means[model] = self.get_resample_data(model_description=model).mean().loc[score_names]

        return resample_means

    def plot_correlation_heatmap(self):
        """
        Creates a plot of the correlations of each of the base-model's predictions on the training set.
        Specifically, cross-validation is used and the predictions from each holdout-fold (for each model) are
            used to build a training set for the stacker. The plot shows the correlations for the predictions
            of each base-model.

        The correlations are taken before the predictions are transformed by the stacker-specific
            transformations.
        """
        if self._model_object is None:
            raise ModelNotFittedError()

        OOLearningHelpers.plot_correlations(correlations=self._train_meta_correlations,
                                            title='Correlations of Models (based on meta-training set)')
        plt.tight_layout()

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def set_persistence_manager(self, persistence_manager):
        """
        NOTE: I need to override this so `train()` in base does not use the persistence_manager passed in
        to this object. I will delegate it manually to the base and stacker models.
        """
        if self._model_object is not None:  # doesn't make sense to configure the cache after we `train()`
            raise ModelAlreadyFittedError()

        self._stackerobject_persistence_manager = persistence_manager

    @staticmethod
    def build_train_meta(data_x, data_y, base_models, scores, converter):
        # build `train_meta` skeleton which will be used to train the stacking_model; we will populate it with
        # the holdout predictions from each fold in the cross-validation (utilizing the Decorator to extract
        # the corresponding holdout predictions and associated indices that will be used to tie the
        # predictions back to the original training set's exact rows)
        resampler_results = list()
        original_indexes = data_x.index.values
        train_meta = pd.DataFrame(index=original_indexes,
                                  columns=[model_info.description for model_info in base_models] +
                                          ['actual_y'])
        train_meta.actual_y = data_y

        # NOTE: may be able to improve performance by utilizing same splits
        # https://github.com/shane-kercheval/oo-learning/issues/1
        # for each base-model, resample the data and get the holdout predictions to build train_meta
        for model_info in base_models:
            decorator = FoldPredictionsDecorator()  # used to extract holdout predictions and indices

            resampler = RepeatedCrossValidationResampler(
                model=model_info.model,
                transformations=model_info.transformations,  # transformations specific to base-model
                scores=[score.clone() for score in scores],
                folds=5,
                repeats=1,
                fold_decorators=[decorator])

            resampler.resample(data_x=data_x, data_y=data_y, hyper_params=model_info.hyper_params)
            resampler_results.append(resampler.results)
            # for regression problems, the predictions will be a numpy array; for classification problems,
            # the predicted values will be a DataFrame with a column per class. For 2 & multi-class problems,
            # we have to figure out how to convert the DF to 1 column/series. So, for example, one option
            # for a two-class problem would be to simply return (via converter) the probabilities associated
            # with the positive class. For a multi-class, it might be to choose the class with the highest
            # probability
            predictions = decorator.holdout_predicted_values
            if isinstance(predictions, pd.DataFrame):
                # if predictions is a DataFrame, then we need a Converter
                assert converter is not None
                predictions = converter.convert(values=predictions)
            else:
                assert isinstance(predictions, np.ndarray) or isinstance(predictions, list)

            assert len(decorator.holdout_predicted_values) == len(original_indexes)
            assert set(train_meta.index.values) == set(original_indexes)
            # noinspection PyTypeChecker
            assert all(train_meta.index.values == original_indexes)

            if isinstance(decorator.holdout_predicted_values, pd.DataFrame):
                # make sure there is a 1-to-1 association with the indexes
                # (although, they won't be in the same order)
                assert set(decorator.holdout_predicted_values.index.values) == set(original_indexes)
            else:
                assert len(predictions) == len(original_indexes)  # predictions same length as training set

            # we need to fill train_meta with the predicted values from the holdout folds
            # however, the indices/predictions from the holdout folds will be in a different order than the
            # original training set. However, we can index off of the holdout_indexes to put the predictions
            # in the right order.
            train_meta.loc[list(decorator.holdout_indexes), model_info.description] = predictions
            train_meta[model_info.description] = train_meta[model_info.description].astype(predictions.dtype)

        return train_meta, resampler_results

    # noinspection PyTypeChecker
    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        # cache the `train_meta` dataset we've built up
        if self._stackerobject_persistence_manager is not None:
            cache = self._stackerobject_persistence_manager.clone()
            # utilizes the cache's _key_prefix if set (e.g. the index of repeat/fold for a resampler).
            # In the case of a Resampler, for example, this ensures a different `train_meta` is cached for
            # each fold
            cache.set_key('train_meta')
            # NOTE: building the train meta dataset is exactly the same regardless of the type of model
            # stacker or the model stacker's hyper-parameters. As such, the file name of the cache will be the
            # same, which will cause problems for multi-threading, because the same file will be read/written
            # simultaneously by multiple threads. Therefore, we will lock this action.
            self.StackerMetaLock().acquire()
            train_meta, self._resampler_results = cache.\
                get_object(fetch_function=lambda: self.build_train_meta(data_x,
                                                                        data_y,
                                                                        self._base_models,
                                                                        self._scores,
                                                                        self._converter))
            self.StackerMetaLock().release()
        else:
            train_meta, self._resampler_results = self.build_train_meta(data_x,
                                                                        data_y,
                                                                        self._base_models,
                                                                        self._scores,
                                                                        self._converter)

        if self._include_original_dataset:
            # need to make sure that the columns aren't overlapping
            assert set(train_meta.columns.values).isdisjoint(set(data_x.columns.values))
            train_meta = train_meta.join(data_x)

        assert all(train_meta.index.values == data_x.index.values)

        # need to fit each base model on ALL of the training data, because when we predict, we first have to
        # build the corresponding train_meta features that we will feed into the trained stacker.
        for model_info in self._base_models:
            transformed_data_x = data_x
            if model_info.transformations:
                # ensure none of the Transformers have been used.
                assert all([x.state is None for x in model_info.transformations])
                # We will fit_transform the training data then in `predict()`,
                # we will transform future data using the same transformations per model
                pipeline = TransformerPipeline(transformations=model_info.transformations)
                # fit on only the train data-set (and also transform)
                transformed_data_x = pipeline.fit_transform(data_x=transformed_data_x)
                # we will reuse the pipelines (which were fitted on the training data) when we predict
                self._base_model_pipelines.append(pipeline)
            else:
                self._base_model_pipelines.append(None)

            if self._stackerobject_persistence_manager is not None:
                cache = self._stackerobject_persistence_manager.clone()
                hyper_params_string = '_'.join(['{}_{}'.format(key, value)
                                                for key, value in model_info.hyper_params.params_dict.items()])  # noqa
                # utilizes the same _key_prefix (e.g. the index of repeat/fold for a resampler). In the case
                # of a Resampler, for example, this ensures a different `train_meta` is cached for each fold
                cache.set_key(key='{}_{}_{}'.format('base', model_info.description, hyper_params_string))
                model_info.model.set_persistence_manager(cache)

            # NOTE: training the base stackers is exactly the same regardless of the type of model
            # stacker or the model stacker's hyper-parameters. As such, the file name of the cache will be the
            # same, which will cause problems for multi-threading, because the same file will be read/written
            # simultaneously by multiple threads
            # Therefore, we will lock this action. This will have the affect of generating performance of
            # training the base models similar to when it is single-threaded, although once created,
            # it should get the benefits of multi-threading. A better version of this would have a separate
            # lock for each base model, so we could still train them simultaneously.
            self.StackerBaseLock().acquire()
            model_info.model.train(data_x=transformed_data_x,
                                   data_y=data_y,
                                   hyper_params=model_info.hyper_params)
            self.StackerBaseLock().release()

        # get the correlations before any transformations on `train_meta` for the stacking model.
        self._train_meta_correlations = train_meta.corr()

        # do stacker-specific transformations
        transformed_train_meta = train_meta.drop(columns='actual_y')
        if self._stacking_model_pipeline is not None:
            transformed_train_meta = self._stacking_model_pipeline.fit_transform(data_x=transformed_train_meta)  # noqa

        if self._train_callback:
            self._train_callback(transformed_train_meta, data_y, hyper_params)

        self._stacking_model.set_persistence_manager(self._stackerobject_persistence_manager)
        self._stacking_model.train(data_x=transformed_train_meta, data_y=data_y, hyper_params=hyper_params)

        return ModelStackerTrainingObject(base_models=self._base_models,
                                          base_model_pipelines=self._base_model_pipelines,
                                          stacking_model=self._stacking_model,
                                          stacking_model_pipeline=self._stacking_model_pipeline)

    # noinspection PyTypeChecker
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        assert isinstance(model_object, ModelStackerTrainingObject)
        original_indexes = data_x.index.values
        # skeleton of test_meta, contains the original indexes and a column for each model
        test_meta = pd.DataFrame(index=original_indexes,
                                 columns=[model_info.description for model_info in model_object.base_models])
        # for each base model, apply applicable transformations, make predictions, and convert predictions to
        # a single column if necessary (i.e. build up predictions used previously fitted pipelines and trained
        # models)
        for index, model_info in enumerate(model_object.base_models):
            # each model will either have an associated Pipeline, or None if no model-specific Transformations
            transformed_data_x = data_x
            pipeline = model_object.base_model_pipelines[index]
            if pipeline:
                transformed_data_x = pipeline.transform(data_x=transformed_data_x)
            assert all(test_meta.index.values == original_indexes)
            predictions = model_info.model.predict(data_x=transformed_data_x)
            if isinstance(predictions, pd.DataFrame):
                # noinspection PyTypeChecker
                assert all(predictions.index.values == original_indexes)

            if self._converter is not None:
                predictions = self._converter.convert(values=predictions)

            test_meta[model_info.description] = predictions  # place predictions in associated column

        if self._include_original_dataset:
            # need to make sure that the columns aren't overlapping
            assert set(test_meta.columns.values).isdisjoint(set(data_x.columns.values))
            test_meta = test_meta.join(data_x)

        assert all(test_meta.index.values == data_x.index.values)

        # now apply any necessary stacker-specific transformations to test_meta
        # for example, some models require center/scaling, which may or may not have been previously done.
        transformed_test_meta = test_meta
        if model_object.stacking_model_pipeline is not None:
            transformed_test_meta = model_object.stacking_model_pipeline.transform(data_x=transformed_test_meta)  # noqa

        if self._predict_callback:
            self._predict_callback(transformed_test_meta)

        return model_object.stacking_model.predict(data_x=transformed_test_meta)
