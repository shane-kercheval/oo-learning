from typing import Union, List

import numpy as np
import pandas as pd

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase
from oolearning.enums.VotingStrategy import VotingStrategy
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class VotingClassifier(ModelWrapperBase):
    """

    """
    def __init__(self,
                 models: List[ModelWrapperBase],
                 voting_strategy: VotingStrategy,
                 converters: Union[List[ContinuousToClassConverterBase], None]=None):
        """
        VotingStrategy.SOFT: for each prediction, averages the model probabilities together, per class
        VotingStrategy.HARD: each class (i.e. column in the returned probabilities) represents the percent of
            votes that each prediction got

            If HARD voting, before each classifier votes, it must transform it's continuous value (e.g.
                probability), to a vote. Since 0.5 is not always the most appropriate value, a converter
                needs to be passed in for each model.
        :param models: pre-trained model, objects
        """
        super().__init__()
        assert len(models) >= 3
        self._models = models
        self._voting_strategy = voting_strategy

        if voting_strategy == VotingStrategy.HARD:
            assert converters is not None
            assert len(converters) == len(models)

        self._converters = converters

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        """
        nothing to do in train(); models are already pre-trained
        """
        return 0

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        model_predictions = [x.predict(data_x=data_x) for x in self._models]

        # need to ensure that all of the resulting prediction dataframes have the same indexes as `data_x`,
        # because we rely on the indexes to calculate the means
        # noinspection PyTypeChecker
        assert all([all(x.index.values == data_x.index.values) for x in model_predictions])

        if self._voting_strategy is VotingStrategy.SOFT:
            df_concat = pd.concat(model_predictions)
            voting_predictions = df_concat.groupby(df_concat.index).mean()
            voting_predictions = voting_predictions.loc[data_x.index.values]

        elif self._voting_strategy is VotingStrategy.HARD:
            classes = list(model_predictions[0].columns.values)

            num_models_converters = len(self._converters)
            class_predictions = [self._converters[x].convert(values=model_predictions[x]) for x in range(0, num_models_converters)]  # noqa
            num_observations = data_x.shape[0]
            # for each class, for each observation, tally the votes for the current class
            # for current_class in classes:
            voting_percentages = [[sum([1 if x[observation_index] == current_class else 0 for x in class_predictions]) / num_models_converters  # noqa
                                   for observation_index in range(0, num_observations)]
                                  for current_class in classes]
            voting_predictions = pd.DataFrame(voting_percentages)
            voting_predictions = voting_predictions.transpose()
            voting_predictions.columns = classes
            voting_predictions.index = data_x.index
        else:
            raise NotImplementedError()

        return voting_predictions
