from abc import ABCMeta, abstractmethod
from typing import List, Callable, Union

import numpy as np
import pandas as pd

from oolearning.model_processors.CloneableFactory import ModelFactory, TransformerFactory, \
    CloneableFactory
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_processors.SingleUseObject import SingleUseObjectMixin
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.transformers.TransformerBase import TransformerBase


class ResamplerBase(SingleUseObjectMixin, metaclass=ABCMeta):
    """
    A Resampler is an object that defines how to 'resample' a data set, for example 'repeated
        cross-validation', and provides information about the performance of the model fit on the resampled
        data.
    """

    def __init__(self,
                 model: ModelWrapperBase,
                 transformations: List[TransformerBase],
                 scores: List[ScoreBase],
                 model_persistence_manager: PersistenceManagerBase = None,
                 results_persistence_manager: PersistenceManagerBase = None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None):
        """
        :param model: (note the model object is not used directly, it is cloned for each resample index)
        :param transformations: (note the Transformer objects are not used directly, they are cloned for each
            resample index)
        :param scores: a `list` of `Evaluator` objects.
            For example, if Kappa and AUC are both score_names of
            interest when resampling, use `holdout_score_objects=[KappaScore, AucRocScore]`;
            if RMSE is the only metric of interest, use `holdout_score_objects=[RMSE]`

            (note the Score objects are not used directly, they are cloned for each
            resample index)

        :param model_persistence_manager: a PersistenceManager defining how the model should be cached,
            optional.
            NOTE: There is currently no enforcement that subclasses of ResamplerBase implement model
            persistence
        :param results_persistence_manager: a PersistenceManager defining how the model should be cached,
            optional.

            If a PersistenceManager is passed in, and the corresponding cached object is found, then
                the resampling procedure is skipped all together.

        :param train_callback: a callback that is called before the model is trained, which returns the
            data_x, data_y, and hyper_params that are passed into the underlying resampler algorithm.
            The primary intent is for unit tests to have the ability to ensure that the data (data_x) is
            being transformed as expected, but it is imaginable to think that users will also benefit
            from this capability to also peak at the data that is being trained.
        """
        super().__init__()

        assert isinstance(model, ModelWrapperBase)
        if transformations is not None:
            assert isinstance(transformations, list)
        assert isinstance(scores, list)

        self._model_factory = ModelFactory(model)
        self._transformer_factory = TransformerFactory(transformations)
        self._score_factory = CloneableFactory(scores)
        self._results = None
        self._model_persistence_manager = model_persistence_manager
        self._results_persistence_manager = results_persistence_manager
        self._train_callback = train_callback
        self._decorators = None

    def set_decorators(self, decorators: List[DecoratorBase]):
        self._decorators = decorators

    @property
    def decorators(self):
        return self._decorators

    @property
    def results(self) -> ResamplerResults:
        if self._results is None:
            raise ModelNotFittedError()

        return self._results

    def append_transformations(self, transformations: List[TransformerBase]):
        self._transformer_factory.append_transformations(transformations=transformations)

    def set_model_persistence_manager(self, persistence_manager: PersistenceManagerBase):
        self._model_persistence_manager = persistence_manager

    def set_results_persistence_manager(self, persistence_manager: PersistenceManagerBase):
        self._results_persistence_manager = persistence_manager

    def resample(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: HyperParamsBase = None):
        """
        `resample()` is friendly name for SingleUseObjectMixin.execute() but both should do the same thing

        `resample` handles the logic of applying the pre-process transformations, as well as fitting the data
            based on teh resampling method and storing the resampled accuracies
        :param data_x: DataFrame to train the model on
        :param data_y: np.ndarray containing the target values to be trained on
        :type hyper_params: object containing the hyper-parameters to tune
        :return: None
        """
        # if we have a persistence manager, grab the cached resampling results if they exist (and skip
        # the resampling)
        self.execute(data_x=data_x, data_y=data_y, hyper_params=hyper_params)

    def _execute(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: HyperParamsBase = None):

        if self._results_persistence_manager:
            self._results = self._results_persistence_manager. \
                get_object(fetch_function=lambda: self._resample(data_x=data_x,
                                                                 data_y=data_y,
                                                                 hyper_params=hyper_params))
        else:
            self._results = self._resample(data_x=data_x, data_y=data_y, hyper_params=hyper_params)

        assert self._results is not None

    def additional_cloning_checks(self):
        pass

    @abstractmethod
    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:
        """
        contains the logic of resampling the data, to be implemented by the sub-class
        :param data_x: is already transformed from the `model_transformations` passed into the constructor
        :param data_y:
        :param hyper_params:
        :return:
        """
        pass
