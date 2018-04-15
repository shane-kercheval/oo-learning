from typing import Union, List, Callable

import numpy as np
import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_processors.RepeatedCrossValidationResampler import RepeatedCrossValidationResampler
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class FoldPredictionsDecorator(DecoratorBase):
    """
    decorator that is passed into the Resampler and extracts the holdout predicted values in order to build
    up `train_meta` in the model stacker.
    """
    def __init__(self):
        self._holdout_indexes = list()
        self._holdout_predicted_values = None

    def decorate(self, **kwargs):
        self._holdout_indexes.extend(kwargs['holdout_indexes'])
        predicted_values = kwargs['holdout_predicted_values']
        # predicted_values is either a DataFrame in the case of classification,
        # or an array in the case of regression
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


class ModelStacker(ModelWrapperBase):
    """
    # TODO: note: the assumption in flow is that each specific model will have been previously cross-validated
        in order to choose the best hyper-params for the specific model.
    adopted from
    http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
    1. Partition the training data into five test folds (note: 5 could be refactored as parameter)
    2. Create a dataset called train_meta with the same row Ids and fold Ids as the training dataset, with
        empty columns M1 and M2.
        Similarly create a dataset called test_meta with the same row Ids as the test dataset and empty
            columns M1 and M2 (NOTE: this will be in the `_predict` function

    3. For each test fold
        3.1 Combine the other four folds to be used as a training fold
        3.2 For each base model (with chosen hyper-params)
        3.2.1 Fit the base model to the training fold and make predictions on the test fold.
            Store these predictions in train_meta to be used as features for the stacking model
            NOTE: i will also have to do the model specific Transformations

    4. Fit each base model to the full training dataset and make predictions on the test dataset.
        Store these predictions inside test_meta
        NOTE: i will make predictions as part of the `_predict` function

    5. Fit a new model, S (i.e the stacking model) to train_meta, using M1 and M2 as features.
        Optionally, include other features from the original training dataset or engineered features.

    6. Use the stacked model S to make final predictions on test_meta
        NOTE: this will be in `_predict`
    """
    def __init__(self,
                 base_models: List[ModelInfo],
                 scores: List[ScoreBase],
                 stacking_model: ModelWrapperBase,
                 stacking_transformations: List[TransformerBase] = None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None,
                 predict_callback: Callable[[pd.DataFrame], None] = None):
        """
        :param base_models:
        :param stacking_transformations: The transformations that are specific to the 'stacker'.
            (The transformations that are specific to the base models will be included in the corresponding
            ModelInfo objects. Transformations applied to ALL models (base and stacker) could be passed in
            via (for example) a ModelFitter.)
        :param scores: since we are cross-validating, we can get a score from each base-model
        """
        super().__init__()
        # ensure unique model descriptions
        assert len(set([x.description for x in base_models])) == len(base_models)
        self._base_models = base_models
        self._scores = scores
        self._stacking_model = stacking_model
        self._stacking_model_pipeline = None if stacking_transformations is None \
            else TransformerPipeline(transformations=stacking_transformations)
        self._resampler_results = list()
        self._base_model_pipelines = list()
        self._train_callback = train_callback
        self._predict_callback = predict_callback
        self._train_meta_correlations = None

    def get_resample_data(self, model_description):
        if self._model_object is None:
            raise ModelNotFittedError()
        model_index = [x.description for x in self._base_models].index(model_description)
        return self._resampler_results[model_index].cross_validation_scores

    def get_resample_means(self):
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
        if self._model_object is None:
            raise ModelNotFittedError()

        OOLearningHelpers.plot_correlations(correlations=self._train_meta_correlations,
                                            title='Correlations of Models (based on meta-training set)')

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        """
        Where going to use the RepeatedCrossValidationResampler, because A) it already takes care of
            model-specific transformations (etc.) and B) we can take advantage of the Fold Decorator
            functionality to get the fold predictions for step `3.2.1`
        :param data_x:
        :param data_y:
        :param hyper_params: hyper-params of the StackingModel
        """
        # build up `train_meta` which will be used to train the stacking_model; we will populate it with the
        # holdout predictions from each fold in the cross-validation (utilizing the Decorator)
        original_indexes = data_x.index.values
        train_meta = pd.DataFrame(index=original_indexes,
                                  columns=[model_info.description for model_info in self._base_models] +
                                          ['actual_y'])
        train_meta.actual_y = data_y

        # NOTE: may be able to improve performance by utilizing same splits
        # https://github.com/shane-kercheval/oo-learning/issues/1
        for model_info in self._base_models:
            decorator = FoldPredictionsDecorator()
            resampler = RepeatedCrossValidationResampler(
                model=model_info.model,
                model_transformations=model_info.transformations,
                scores=[score.clone() for score in self._scores],
                folds=5,
                repeats=1,
                fold_decorators=[decorator])
            resampler.resample(data_x=data_x, data_y=data_y, hyper_params=model_info.hyper_params)
            self._resampler_results.append(resampler.results)
            # the predicted values will be a dataframe with a column per class. For 2 & multi-class problems,
            # we have to figure out how to convert the DF to 1 column/series. So, for example, one option
            # for a two-class problem would be to simply return (via converter) the probabilities associated
            # with the positive class. For a multi-class, it might be to choose the class with the highest
            # probability
            predictions = decorator.holdout_predicted_values
            if model_info.converter is not None:
                predictions = model_info.converter.convert(values=predictions)

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
            # however, the holdout folds will be in a different order
            # get the indexes (in the order that the decorator has them in, which will be different than
            # the order they were originally in) and fill the necessary column with the predictions
            train_meta.loc[list(decorator.holdout_indexes), model_info.description] = predictions
            train_meta[model_info.description] = train_meta[model_info.description].astype(predictions.dtype)

            transformed_data_x = data_x
            if model_info.transformations:
                # ensure none of the Transformers have been used. We will fit_transform the training data
                # then in `predict()`, we will transform future data using the same transformations per model
                assert all([x.state is None for x in model_info.transformations])
                pipeline = TransformerPipeline(transformations=model_info.transformations)  # noqa
                # fit on only the train dataset (and also transform)
                transformed_data_x = pipeline.fit_transform(data_x=data_x)
                self._base_model_pipelines.append(pipeline)
            else:
                self._base_model_pipelines.append(None)

            model_info.model.train(data_x=transformed_data_x,
                                   data_y=data_y,
                                   hyper_params=model_info.hyper_params)

        # get the correlations before any transformations on `train_meta` for the stacking model.
        self._train_meta_correlations = train_meta.corr()

        transformed_train_meta = train_meta
        if self._stacking_model_pipeline is not None:
            # noinspection PyTypeChecker
            transformed_train_meta = self._stacking_model_pipeline.fit_transform(data_x=transformed_train_meta.drop(columns='actual_y'))  # noqa
            transformed_train_meta['actual_y'] = data_y

        if self._train_callback:
            self._train_callback(transformed_train_meta.drop(columns='actual_y'),
                                 transformed_train_meta.actual_y,
                                 hyper_params)
        # noinspection PyTypeChecker
        self._stacking_model.train(data_x=transformed_train_meta.drop(columns='actual_y'),
                                   data_y=transformed_train_meta.actual_y,
                                   hyper_params=hyper_params)
        return 'none'

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        original_indexes = data_x.index.values
        # skeleton of test_meta, contains the original indexes and a column for each model
        test_meta = pd.DataFrame(index=original_indexes,
                                 columns=[model_info.description for model_info in self._base_models])
        # for each model, apply applicable transformations, make predictions, and convert predictions to
        # a single column if necessary
        for index, model_info in enumerate(self._base_models):
            # each model will either have an associated Pipeline, or None if no model-specific Transformations
            transformed_data_x = data_x
            pipeline = self._base_model_pipelines[index]
            if pipeline:
                transformed_data_x = pipeline.transform(data_x=data_x)
            # noinspection PyTypeChecker
            assert all(test_meta.index.values == original_indexes)
            predictions_raw = model_info.model.predict(data_x=transformed_data_x)
            if isinstance(predictions_raw, pd.DataFrame):
                # noinspection PyTypeChecker
                assert all(predictions_raw.index.values == original_indexes)

            predictions = predictions_raw
            if model_info.converter is not None:
                predictions = model_info.converter.convert(values=predictions)

            test_meta[model_info.description] = predictions  # place predictions in associated column

        # now apply any necessary transformations to test_meta
        # for example, some models require center/scaling, which may or may not have been previously done.
        transformed_test_meta = test_meta
        if self._stacking_model_pipeline is not None:
            transformed_test_meta = self._stacking_model_pipeline.transform(data_x=transformed_test_meta)

        if self._predict_callback:
            self._predict_callback(transformed_test_meta)

        return self._stacking_model.predict(data_x=transformed_test_meta)
