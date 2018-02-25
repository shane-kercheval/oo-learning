from typing import List, Callable, Union

import numpy as np
import pandas as pd

from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.StatelessTransformer import StatelessTransformer
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class RepeatedCrossValidationResampler(ResamplerBase):
    """
    Traditional k-fold repeated cross validation. Does NOT stratify data based on target values.
    """
    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase],
                 scores: List[ScoreBase],
                 persistence_manager: PersistenceManagerBase = None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None,
                 folds=5,
                 repeats=5):
        super().__init__(model=model,
                         model_transformations=model_transformations,
                         scores=scores,
                         persistence_manager=persistence_manager,
                         train_callback=train_callback)

        assert isinstance(folds, int)
        assert isinstance(repeats, int)

        self._folds = folds
        self._repeats = repeats


# TODO document that transformations are fit/transformed on the training folds and transformed on the holdout
    # fold; to avoid 'snooping'
    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:

        result_scores = list()  # list of all the `evaluated` holdout scores

        # transform/fit on training data
        if self._model_transformations is not None:
            # before we fit the data, we actually want to 'snoop' at what the expected columns will be with
            # ALL the data. The reason is that if we so some sort of encoding (dummy/one-hot), but not all of
            # the categories are included in the training set (i.e. maybe only a small number of observations
            # have the categoric value), then we can still ensure that we will be giving the same expected
            # columns/encodings to the `predict` method with the holdout set.
            expected_columns = TransformerPipeline.get_expected_columns(data=data_x,
                                                                        transformations=self._model_transformations)  # noqa
            # create a transformer that ensures the expected columns exist, and add it as the last
            # transformation
            transformer = StatelessTransformer(custom_function=lambda x_df: x_df.reindex(columns=expected_columns,  # noqa
                                                                                         fill_value=0))
            self._model_transformations = self._model_transformations + [transformer]

        for repeat_index in range(self._repeats):
            # consistent folds per repeat index
            np.random.seed(repeat_index)
            # generate random fold #s that correspond to each index of the data
            random_folds = np.random.randint(low=0, high=self._folds, size=len(data_y))

            for fold_index in range(self._folds):  # for each fold, train and calculate

                holdout_indexes = random_folds == fold_index  # indexes that match the fold belong to holdout
                training_indexes = ~holdout_indexes  # all other indexes belong to the training set

                # odd naming serves as distinction between when i'm using transformed/non-transformed data
                train_x_not_transformed, holdout_x_not_transformed = data_x[training_indexes], data_x[holdout_indexes]  # noqa
                train_y, holdout_y = data_y[training_indexes], data_y[holdout_indexes]

                # noinspection PyTypeChecker
                pipeline = TransformerPipeline(transformations=None if self._model_transformations is None else [x.clone() for x in self._model_transformations])  # noqa
                train_x_transformed = pipeline.fit_transform(data_x=train_x_not_transformed)
                holdout_x_transformed = pipeline.transform(data_x=holdout_x_not_transformed)

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
                fold_scores = list()
                for score in self._scores:
                    score_copy = score.clone()  # need to reuse this object type for each fold/repeat
                    score_copy.calculate(actual_values=holdout_y,
                                         predicted_values=model_copy.predict(data_x=holdout_x_transformed))
                    fold_scores.append(score_copy)
                result_scores.append(fold_scores)
        # result_scores is a list of list of holdout scores.
        # Each outer list represents a resampling result
        # and each element of the inner list represents a specific score.
        return ResamplerResults(scores=result_scores)

    @staticmethod
    def build_cache_key(model: ModelWrapperBase,
                        hyper_params: HyperParamsBase,
                        repeat_index: int,
                        fold_index: int) -> str:
        model_name = type(model).__name__
        if hyper_params is None:
            key = model_name
        else:
            # if hyper-params, flatten out list of param names and values and concatenate/join them together
            hyper_params_long = '_'.join([str(x) + str(y) for x, y in hyper_params.params_dict.items()])
            key = '_'.join(['repeat' + str(repeat_index), 'fold' + str(fold_index), model_name, hyper_params_long])  # noqa

        return key
