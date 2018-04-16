from typing import Union

import numpy as np
import pandas as pd

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class MockRegressionModelWrapperTrainingObject:
    def __init__(self, model_object, target_probabilities, target_intervals):
        self._model_object = model_object
        self._target_probabilities = target_probabilities
        self._target_intervals = target_intervals


class MockRegressionModelWrapper(ModelWrapperBase):

    @property
    def feature_importance(self):
        raise NotImplementedError()

    @property
    def results_summary(self) -> object:
        return 'test_summary'

    def __init__(self, data_y: np.ndarray, model_object: str='test model_object'):
        """
        Dumb mock object that randomly returns values corresponding with a similar distribution as `data_y`
        :type model_object: string that can be used to ensure the correct model_object is returned
        :param data_y: actual values, used to know which values to randomly pass back in `predict()`
        """
        super().__init__()
        self._model_object_pre_train = model_object
        self.fitted_train_x = None

        if not isinstance(data_y, pd.Series):
            data_y = pd.Series(data_y)

        # gets the distribution of unique values, unique values being .index.values
        value_distributions = data_y.value_counts(normalize=True, bins=10)
        self._target_intervals = value_distributions.index.values.tolist()
        self._target_probabilities = value_distributions.values.tolist()

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        self.fitted_train_x = data_x
        self._model_object = self._model_object_pre_train
        return MockRegressionModelWrapperTrainingObject(model_object=self._model_object,
                                                        target_probabilities=self._target_probabilities,
                                                        target_intervals=self._target_intervals)

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        # get length of data, return random
        np.random.seed(123)
        # generate random `0` through `(len-1)` following the distribution found in data_y,
        # generate n=len(data_x) predictions
        random_predictions = np.random.choice(a=np.arange(0, len(model_object._target_probabilities)),
                                              p=model_object._target_probabilities,
                                              size=len(data_x))
        # pd.Series(random_predictions).value_counts(normalize=True)

        # the generated numbers should be random floats correspond to the associated interval;
        # inclusive/exclusive won't exactly match but it doesn't matter for this.
        # this will only return unique values for each interval because of the seed; again, doesn't matter
        def get_random_float(interval: pd.Interval) -> float:
            np.random.seed(123)
            return round(np.random.uniform(low=interval.left, high=interval.right, size=1)[0], 1)

        return np.array([get_random_float(interval=model_object._target_intervals[x])
                         for x in random_predictions])
