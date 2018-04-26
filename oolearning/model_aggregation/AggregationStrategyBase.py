from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd


class AggregationStrategyBase(metaclass=ABCMeta):
    """
    Aggregators combine the predictions of various models into a single prediction.
    """
    @abstractmethod
    def aggregate(self, model_predictions: List[Union[pd.DataFrame, np.ndarray]]) \
            -> Union[np.ndarray, pd.DataFrame]:
        """
        Defines how the model predictions will be aggregated together.

        :param model_predictions: each list item will be the model predictions (either a `DataFrame` of
            predictions per model for classification problems, or a `np.ndarray` in the case of regressions
            problems.
        :return: `Dataframe` of continuous values (e.g. probabilities) for Classification Problems (1 column
            per class); or, for Regression problems, an `ndarray` with the aggregated values.
        """
        pass
