from typing import List, Callable, Union

import numpy as np
import pandas as pd

from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.transformers.StatelessTransformer import StatelessTransformer
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class RepeatedCrossValidationResampler(ResamplerBase):
    """
    Traditional k-fold repeated cross validation. Does NOT stratify data based on target values.
    """
    def __init__(self,
                 model: ModelWrapperBase,
                 transformations: Union[List[TransformerBase], None],
                 scores: List[ScoreBase],
                 persistence_manager: PersistenceManagerBase = None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None,
                 folds: int=5,
                 repeats: int=5,
                 fold_decorators: List[DecoratorBase]=None):
        """
        :param model: The model to fit at each fold. A clone/copy of the model is created at each fold.
        :param transformations: The transformations that are to be applied at each fold. Specifically,
            the transformations are fit/transformed on the training folds, and then transformed
            (without being fit) on the holdout set. The objects are cloned/copied at each fold.
        :param scores: the Scores that are evaluated at each fold/repeat.
        :param persistence_manager: an object describing how to save/cache the trained models.
        :param train_callback: a callback that is called at each fold after the transformations and
            before the model is trained. Its primary use is originally for testing and data
            verification.
        :param folds: the number of folds to divide the data-set into
        :param repeats: The number of times to repeat the procedure. For each repeat, a different
            random seed is used to create different fold indexes.
        :param fold_decorators: intent is to add responsibility the Resampler dynamically. This decorator is
            called at the end of each fold and is passed the `scores`, the holdout actual values, and the
            holdout predicted values.
        """
        super().__init__(model=model,
                         transformations=transformations,
                         scores=scores,
                         persistence_manager=persistence_manager,
                         train_callback=train_callback)

        assert isinstance(folds, int)
        assert isinstance(repeats, int)

        self._folds = folds
        self._repeats = repeats
        self._decorators = fold_decorators

    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:

        result_scores = list()  # list of all the `evaluated` holdout scores

        # transform/fit on training data
        if self._transformations is not None:
            # before we fit the data, we actually want to 'snoop' at what the expected columns will be with
            # ALL the data. The reason is that if we so some sort of encoding (dummy/one-hot), but not all of
            # the categories are included in the training set (i.e. maybe only a small number of observations
            # have the categoric value), then we can still ensure that we will be giving the same expected
            # columns/encodings to the `predict` method of the holdout set.
            expected_columns = TransformerPipeline.get_expected_columns(data=data_x,
                                                                        transformations=self._transformations)  # noqa
            # create a transformer that ensures the expected columns exist (e.g. dummy columns), and add it
            # as the last transformation
            transformer = StatelessTransformer(custom_function=lambda x_df: x_df.reindex(columns=expected_columns,  # noqa
                                                                                         fill_value=0))
            self._transformations = self._transformations + [transformer]

        for repeat_index in range(self._repeats):
            # consistent folds per repeat index, but different folds for different repeats
            np.random.seed(repeat_index)
            # generate random fold #s that correspond to each index of the data
            random_folds = np.random.randint(low=0, high=self._folds, size=len(data_y))

            for fold_index in range(self._folds):

                holdout_indexes = random_folds == fold_index  # indexes that match the fold belong to holdout
                training_indexes = ~holdout_indexes  # all other indexes belong to the training set

                # odd naming serves as distinction between when we use transformed/non-transformed data
                train_x_not_transformed, holdout_x_not_transformed = data_x[training_indexes],\
                                                                     data_x[holdout_indexes]
                train_y, holdout_y = data_y[training_indexes], data_y[holdout_indexes]

                # NOTE: we are fitting the transformations on the k-1 folds (i.e. local training data)
                # for each k times we train/predict data. This is so we don't have any contamination/
                # leakage into the local holdout/fold we are predicting on (just like we wouldn't fit
                # the transformations on the entire dataset; we fit/transform on the training and then
                # simply transform on the holdout
                pipeline = TransformerPipeline(transformations=None if self._transformations is None else [x.clone() for x in self._transformations])  # noqa
                # before we fit the data, we actually want to 'peak' at what the expected columns will be with
                # ALL the data. The reason is that if we so some sort of encoding (dummy/one-hot), but not all
                # of the categories are included in the training set (i.e. maybe only a small number of
                # observations have the categoric value), then we can still ensure that we will be giving the
                # same expected columns/encodings to the `predict` method with the holdout set.
                # peak at all the data
                pipeline.peak(data_x=data_x)
                # fit on only the train dataset (and also transform)
                train_x_transformed = pipeline.fit_transform(data_x=train_x_not_transformed)
                # transform (but don't fit) on holdout
                holdout_x_transformed = pipeline.transform(data_x=holdout_x_not_transformed)

                # the callback allows callers to see/verify the data that is being trained, at each fold
                if self._train_callback is not None:
                    self._train_callback(train_x_transformed, data_y, hyper_params)

                model_copy = self._model.clone()  # need to reuse this object type for each fold/repeat

                # set up persistence if applicable
                if self._persistence_manager is not None:  # then build the key
                    cache_key = self.build_cache_key(model=model_copy,
                                                     hyper_params=hyper_params,
                                                     repeat_index=repeat_index,
                                                     fold_index=fold_index)
                    self._persistence_manager.set_key(key=cache_key)
                    model_copy.set_persistence_manager(persistence_manager=self._persistence_manager)

                model_copy.train(data_x=train_x_transformed, data_y=train_y, hyper_params=hyper_params)
                predicted_values = model_copy.predict(data_x=holdout_x_transformed)

                fold_scores = list()
                for score in self._scores:  # cycle through scores and store results of each fold
                    score_copy = score.clone()  # need to reuse this object type for each fold/repeat
                    score_copy.calculate(actual_values=holdout_y,
                                         predicted_values=predicted_values)
                    fold_scores.append(score_copy)
                result_scores.append(fold_scores)

                # executed any functionality that is dynamically attached via decorators
                if self._decorators:
                    for decorator in self._decorators:
                        decorator.decorate(repeat_index=repeat_index,
                                           fold_index=fold_index,
                                           scores=self._scores,
                                           holdout_actual_values=holdout_y,
                                           holdout_predicted_values=predicted_values,
                                           holdout_indexes=holdout_x_transformed.index.values)

        # result_scores is a list of list of holdout scores.
        # Each outer list represents a resampling result
        # and each element of the inner list represents a specific score.
        return ResamplerResults(scores=result_scores, decorators=self._decorators)

    @staticmethod
    def build_cache_key(model: ModelWrapperBase,
                        hyper_params: HyperParamsBase,
                        repeat_index: int,
                        fold_index: int) -> str:
        """
        :return: returns a key that acts as, for example, the file name of the model being cached for the
            persistence manager; has the form:
                `repeat[repeat number]_fold[fold number]_[Model Class Name]_[hyper param values]`
        """
        model_name = model.name
        if hyper_params is None:
            key = model_name
        else:
            # if hyper-params, flatten out list of param names and values and concatenate/join them together
            hyper_params_long = '_'.join([str(x) + str(y) for x, y in hyper_params.params_dict.items()])
            key = '_'.join(['repeat' + str(repeat_index), 'fold' + str(fold_index), model_name, hyper_params_long])  # noqa

        return key
