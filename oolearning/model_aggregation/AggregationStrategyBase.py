from abc import ABCMeta, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd


class AggregationStrategyBase(metaclass=ABCMeta):
    @abstractmethod
    def aggregate(self, model_predictions: List[Union[pd.DataFrame, np.ndarray]]) -> Union[np.ndarray, pd.DataFrame]:  # noqa
        """
        :param model_predictions: these will be the model predictions from the models passed into the
            `ModelAggregator`. `model_predictions` is a list, with the predictions from each model represented
            in a single list index. The object in the list index will either be a `pd.DataFrame` in the
            case of a classification problem or an `np.ndarray` in the case of a regression problem.
        :return: Dataframe of continuous values (e.g. probabilities) for Classification Problems (1 column
            per class)
            Or, for Regression problems, an ndarray with the aggregated values.
        """
        pass
