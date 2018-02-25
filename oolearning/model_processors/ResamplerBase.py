from abc import ABCMeta, abstractmethod
from typing import List, Callable, Union

import copy
import pandas as pd
import numpy as np

from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class ResamplerBase(metaclass=ABCMeta):
    """
    A Resampler is an object that defines how to 'resample' a data set, for example 'repeated
        cross-validation'
    """

    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase],
                 scores: List[ScoreBase],
                 persistence_manager: PersistenceManagerBase = None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None):
        """
        :param model:
        :param model_transformations:
        :param scores: a `list` of `Evaluator` objects.
            For example, if Kappa and AUC are both metrics of
            interest when resampling, use `holdout_scores=[KappaScore, AucScore]`;
            if RMSE is the only metric of interest, use `holdout_scores=[RMSE]`
        :param persistence_manager: a PersistenceManager defining how the model should be cached, optional.
            NOTE: There is currently no enforcement that subclasses of ResamplerBase implement model
            persistence
        :param train_callback: a callback that is called before the model is trained, which returns the
            data_x, data_y, and hyper_params that are passed into the underlying resampler algorithm.
            The primary intent is for unit tests to have the ability to ensure that the data (data_x) is
            being transformed as expected, but it is imaginable to think that users will also benefit
            from this capability to also peak at the data that is being trained.
        """
        assert isinstance(model, ModelWrapperBase)
        if model_transformations is not None:
            assert isinstance(model_transformations, list)
        assert isinstance(scores, list)

        self._model = model
        self._model_transformations = model_transformations
        self._scores = scores
        self._results = None
        self._persistence_manager = persistence_manager
        self._train_callback = train_callback

    def clone(self):
        """
        when, for example, tuning, an Resampler will have to be cloned several times (before using)
        :return: a clone of the current object
        """
        assert self._results is None  # only intended on being called before evaluating
        return copy.deepcopy(self)

    @property
    def results(self) -> ResamplerResults:
        if self._results is None:
            raise ModelNotFittedError()

        return self._results

    def set_persistence_manager(self, persistence_manager: PersistenceManagerBase):
        self._persistence_manager = persistence_manager

    def resample(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: HyperParamsBase = None):
        """
        `resample` handles the logic of applying the pre-process transformations, as well as fitting the data
            based on teh resampling method and storing the resampled accuracies
        :param data_x: DataFrame to fit the model on
        :param data_y: np.ndarray containing the target values to be trained on
        :type hyper_params: object containing the hyper-parameters to tune
        :return: None
        """
        self._results = self._resample(data_x=data_x, data_y=data_y, hyper_params=hyper_params)

    @abstractmethod
    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase=None) -> ResamplerResults:
        """
        contains the logic of resampling the data, to be implemented by the sub-class
        :param data_x: is already transformed from the `model_transformations` passed into the constructor
        :param data_y:
        :param hyper_params:
        :return:
        """
        pass
