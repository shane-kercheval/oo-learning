from typing import List, Union

import numpy as np
import pandas as pd

from oolearning import ContinuousToClassConverterBase
from oolearning.model_aggregation.AggregationStrategyBase import AggregationStrategyBase


class HardVotingAggregationStrategy(AggregationStrategyBase):
    """
    Voting strategy for classification problems.

    Before the vote, the predictions are transformed from it's continuous value (e.g. probability), to a
    specific class prediction. The percent of votes is recorded for each class (i.e. column).

    Since 0.5 is not always the most appropriate value, a converter needs to be passed in for each model.
    """
    def __init__(self, converters: List[ContinuousToClassConverterBase]):
        super().__init__()
        self._converters = converters

    def aggregate(self, model_predictions: List[Union[pd.DataFrame, np.ndarray]]) \
            -> Union[np.ndarray, pd.DataFrame]:
        assert isinstance(model_predictions, list)
        assert all([isinstance(x, pd.DataFrame) for x in model_predictions])
        assert len(self._converters) == len(model_predictions)
        classes = list(model_predictions[0].columns.values)

        num_models_converters = len(self._converters)
        class_predictions = [self._converters[x].convert(values=model_predictions[x]) for x in
                             range(0, num_models_converters)]  # noqa
        num_observations = model_predictions[0].shape[0]
        # for each class, for each observation, tally the votes for the current class
        # for current_class in classes:
        voting_percentages = [[sum([1 if x[observation_index] == current_class else 0 for x in
                                    class_predictions]) / num_models_converters  # noqa
                               for observation_index in range(0, num_observations)]
                              for current_class in classes]
        voting_predictions = pd.DataFrame(voting_percentages)
        voting_predictions = voting_predictions.transpose()
        voting_predictions.columns = classes
        voting_predictions.index = model_predictions[0].index

        return voting_predictions
