from typing import Union, List

import numpy as np
import pandas as pd

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_processors.RepeatedCrossValidationResampler import RepeatedCrossValidationResampler
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class FoldPredictionsDecorator(DecoratorBase):
    def __init__(self):
        self._holdout_indexes = list()
        self._holdout_predicted_values = pd.DataFrame()

    def decorate(self, **kwargs):
        self._holdout_indexes.extend(kwargs['holdout_indexes'])
        self._holdout_predicted_values = self._holdout_predicted_values.append(kwargs['holdout_predicted_values'])  # noqa

    @property
    def holdout_indexes(self):
        return self._holdout_indexes

    @property
    def holdout_predicted_values(self):
        return self._holdout_predicted_values


class TwoClassExtractPositivePredictions(TwoClassConverterBase):

    def convert(self, values: pd.DataFrame) -> np.ndarray:
        return values[self.positive_class]


class ModelStacker(ModelWrapperBase):
    def __init__(self,
                 base_models: List[ModelInfo],
                 scores: List[ScoreBase],
                 stacking_model: ModelWrapperBase):
        """
        :param base_models:
        :param scores: since we are cross-validating, we can get a score from each base-model

        """
        super().__init__()
        self._base_models = base_models
        self._scores = scores
        self._stacking_model = stacking_model
        self._resampler_results = list()

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
            predictions = model_info.converter.convert(values=decorator.holdout_predicted_values)

            if isinstance(decorator.holdout_predicted_values, pd.DataFrame):
                assert len(decorator.holdout_predicted_values) == len(original_indexes)
                # make sure there is a 1-to-1 association with the indexes
                # (although, they won't be in the same order)
                assert set(decorator.holdout_predicted_values.index.values) == set(original_indexes)
                assert set(train_meta.index.values) == set(original_indexes)
                # noinspection PyTypeChecker
                assert all(train_meta.index.values == original_indexes)

                # we need to fill train_meta with the predicted values from the holdout folds
                # however, the holdout folds will be in a different order
                # get the indexes (in the order that the decorator has them in, which will be different than
                # the order they were originally in) and fill the necessary column with the predictions
                train_meta.loc[list(decorator.holdout_indexes), model_info.description] = predictions

                pd.crosstab(train_meta.cart, train_meta.actual_y, rownames=['a'], colnames=['b'])
                pd.crosstab([1 if x > 0.5 else 0 for x in train_meta.random_forest.values], train_meta.actual_y, rownames=['a'], colnames=['b'])
            else:
                raise NotImplementedError()

            model_info.model.train(data_x=data_x,
                                   data_y=data_y,
                                   hyper_params=model_info.hyper_params)

        # noinspection PyTypeChecker
        self._stacking_model.train(data_x=train_meta.drop(columns='actual_y'),
                                   data_y=train_meta.actual_y,
                                   hyper_params=hyper_params)
        return 'none'

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        original_indexes = data_x.index.values
        test_meta = pd.DataFrame(index=original_indexes,
                                 columns=[model_info.description for model_info in self._base_models])
        for model_info in self._base_models:
            predictions = model_info.model.predict(data_x=data_x)

            test_meta[model_info.description] = predictions

        return self._stacking_model.predict(data_x=test_meta)

# TODO: note: the assumption in flow is that each specific model will have been previously cross-validated in
    # order to choose the best hyper-params for the specific model.
# TODO: note... Model specific Transformations might be tricky...
#   A) model transformations are typically handled by some sort of "processor" like fitter or tuner
#   B) each model has it's own model-specific transformation(s). (e.g. some models should be centered/scaled,
    # while others do not need to (or should not) be.
# (Suppose we use ModelFitter); The `data_x` passed in is the full training set. We will assume a test set
# is withheld.
# adopted from
# http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
# 1. Partition the training data into five test folds (note: 5 could be refactored as parameter)
# 2. Create a dataset called train_meta with the same row Ids and fold Ids as the training dataset, with empty
    # columns M1 and M2.
    # Similarly create a dataset called test_meta with the same row Ids as the test dataset and empty columns
        # M1 and M2 (NOTE: this will be in the `_predict` function

# 3. For each test fold
    # 3.1 Combine the other four folds to be used as a training fold
    # 3.2 For each base model (with chosen hyper-params)
    # 3.2.1 Fit the base model to the training fold and make predictions on the test fold.
        # Store these predictions in train_meta to be used as features for the stacking model
        # NOTE: i will also have to do the model specific Transformations

# 4. Fit each base model to the full training dataset and make predictions on the test dataset.
    # Store these predictions inside test_meta
    # NOTE: i will make predictions as part of the `_predict` function

# 5. Fit a new model, S (i.e the stacking model) to train_meta, using M1 and M2 as features.
    # Optionally, include other features from the original training dataset or engineered features.

# 6. Use the stacked model S to make final predictions on test_meta
    # NOTE: this will be in `_predict`