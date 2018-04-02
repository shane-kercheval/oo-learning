from typing import List, Union

import numpy as np
import pandas as pd

from oolearning.model_aggregation.AggregationStrategyBase import AggregationStrategyBase


class SoftVotingAggregationStrategy(AggregationStrategyBase):
    """
    Voting strategy for classification problems. For each prediction, averages the model probabilities
        together, per class
    """

    def aggregate(self, model_predictions: List[Union[pd.DataFrame, np.ndarray]]) -> \
            Union[np.ndarray, pd.DataFrame]:
        assert isinstance(model_predictions, list)
        assert isinstance(model_predictions[0], pd.DataFrame)
        df_concat = pd.concat(model_predictions)
        voting_predictions = df_concat.groupby(df_concat.index).mean()
        voting_predictions = voting_predictions.loc[model_predictions[0].index.values]

        return voting_predictions
