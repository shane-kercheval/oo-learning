from typing import List, Union

import numpy as np
import pandas as pd

from oolearning import AggregationStrategyBase


class MedianAggregationStrategy(AggregationStrategyBase):
    """
    Voting strategy for regression problems. Takes the median of all the models' predictions.
    """

    def aggregate(self, model_predictions: List[Union[pd.DataFrame, np.ndarray]]) -> \
            Union[np.ndarray, pd.DataFrame]:
        assert isinstance(model_predictions, list)
        assert all([isinstance(x, np.ndarray) for x in model_predictions])

        return np.asarray([np.median(x) for x in zip(*model_predictions)])
