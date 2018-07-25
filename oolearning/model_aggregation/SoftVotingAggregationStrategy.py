from typing import List, Union

import numpy as np
import pandas as pd

from oolearning.model_aggregation.AggregationStrategyBase import AggregationStrategyBase


class SoftVotingAggregationStrategy(AggregationStrategyBase):
    """
    Voting strategy for classification problems. All of the prediction DataFrames are averaged together.
    """
    def __init__(self, aggregation: callable=np.mean):
        self._aggregation = aggregation

    def aggregate(self, model_predictions: List[Union[pd.DataFrame, np.ndarray]]) -> \
            Union[np.ndarray, pd.DataFrame]:
        assert isinstance(model_predictions, list)
        assert all([isinstance(x, pd.DataFrame) for x in model_predictions])
        df_concat = pd.concat(model_predictions)
        voting_predictions = df_concat.groupby(df_concat.index).aggregate(lambda x: self._aggregation(x))
        voting_predictions = voting_predictions.loc[model_predictions[0].index.values]

        return voting_predictions
