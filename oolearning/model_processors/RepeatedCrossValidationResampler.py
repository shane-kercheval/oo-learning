from typing import List

import numpy as np
import pandas as pd

from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase


class RepeatedCrossValidationResampler(ResamplerBase):
    """
    Traditional k-fold repeated cross validation. Does NOT stratify data based on target values.
    """
    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase],
                 evaluators: List[EvaluatorBase],
                 persistence_manager: PersistenceManagerBase = None,
                 folds=5,
                 repeats=5):
        super().__init__(model=model,
                         model_transformations=model_transformations,
                         evaluators=evaluators,
                         persistence_manager=persistence_manager)

        assert isinstance(folds, int)
        assert isinstance(repeats, int)

        self._folds = folds
        self._repeats = repeats

    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:

        result_evaluators = list()  # list of all the `evaluated` holdout_evaluators
        for repeat_index in range(self._repeats):
            # consistent folds per repeat index
            np.random.seed(repeat_index)
            # generate random fold #s that correspond to each index of the data
            random_folds = np.random.randint(low=0, high=self._folds, size=len(data_y))

            for fold_index in range(self._folds):  # for each fold, train and evaluate

                testing_indexes = random_folds == fold_index  # indexes matching the fold belong to test set
                training_indexes = ~testing_indexes  # all other indexes belongs to the training set

                train_x, test_x = data_x[training_indexes], data_x[testing_indexes]
                train_y, test_y = data_y[training_indexes], data_y[testing_indexes]

                model_copy = self._model.clone()  # need to reuse this object type for each fold/repeat

                # set up persistence if applicable
                if self._persistence_manager is not None:  # then build the key
                    cache_key = self.build_cache_key(model=model_copy,
                                                     hyper_params=hyper_params,
                                                     repeat_index=repeat_index,
                                                     fold_index=fold_index)
                    self._persistence_manager.set_key(key=cache_key)
                    model_copy.set_persistence_manager(persistence_manager=self._persistence_manager)

                model_copy.train(data_x=train_x, data_y=train_y, hyper_params=hyper_params)
                fold_evaluators = list()
                for evaluator in self._evaluators:
                    evaluator_copy = evaluator.clone()  # need to reuse this object type for each fold/repeat
                    evaluator_copy.evaluate(actual_values=test_y,
                                            predicted_values=model_copy.predict(data_x=test_x))
                    fold_evaluators.append(evaluator_copy)
                result_evaluators.append(fold_evaluators)
        # result_evaluators is a list of list of holdout_evaluators.
        # Each outer list represents a resampling result
        # and each element of the inner list represents a specific evaluator.
        return ResamplerResults(evaluators=result_evaluators)

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
