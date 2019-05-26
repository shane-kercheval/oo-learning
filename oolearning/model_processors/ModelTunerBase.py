import time
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd

from oolearning.model_processors.TunerResultsBase import TunerResultsBase
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError


class ModelTunerBase(metaclass=ABCMeta):
    """
    A GridSearchModelTuner searches for the best hyper-parameters. A specific type of GridSearchModelTuner might use "Grid Search"
        as a brute force method of searching for the best parameters. Another type of GridSearchModelTuner might use
        Bayesian Optimization.

        In each case, the GridSearchModelTuner will return the optimal parameters for the corresponding constraints.
    """
    def __init__(self):
        self._results = None
        self._total_tune_time = None

    @property
    def results(self) -> TunerResultsBase:
        """
        :return: the results as a TunerResultsBase object
        """
        if self._results is None:
            raise ModelNotFittedError()

        return self._results

    @property
    def total_tune_time(self) -> float:
        """
        :return: total time
        """
        if self._results is None:
            raise ModelNotFittedError()

        return self._total_tune_time

    def tune(self, data_x: pd.DataFrame, data_y: np.ndarray):
        start_time = time.time()
        self._results = self._tune(data_x=data_x, data_y=data_y)
        self._total_tune_time = time.time() - start_time

    @abstractmethod
    def _tune(self, data_x: pd.DataFrame, data_y: np.ndarray) -> TunerResultsBase:
        """
        contains the logic of tuning the data, to be implemented by the sub-class
        :param data_x:
        :param data_y:
        :return: TunerResultsBase
        """
        pass
