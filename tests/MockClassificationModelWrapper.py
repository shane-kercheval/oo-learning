from typing import Union

import numpy as np
import pandas as pd

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class MockRegressionModelWrapperTrainingObject:
    def __init__(self, model_object, target_probabilities, unique_targets):
        self._model_object = model_object
        self._target_probabilities = target_probabilities
        self._unique_targets = unique_targets


class MockClassificationModelWrapper(ModelWrapperBase):
    @property
    def feature_importance(self):
        raise NotImplementedError()

    def __init__(self, data_y: np.ndarray):
        """
        Dumb mock object that randomly returns values of the target class (i.e. unique data_y values)
        :param data_y: actual values, used to know which values to randomly pass back in `predict()`
        """
        super().__init__()
        self.fitted_train_x = None

        if not isinstance(data_y, pd.Series):
            data_y = pd.Series(data_y)

        # gets the distribution of unique values, unique values being .index.values
        value_distributions = data_y.value_counts(normalize=True)
        self._unique_targets = value_distributions.index.values.tolist()
        self._target_probabilities = value_distributions.values.tolist()

    @property
    def results_summary(self) -> object:
        return 'test_summary'

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        self.fitted_train_x = data_x
        self._model_object = 'junk'
        return MockRegressionModelWrapperTrainingObject(model_object='test',
                                                        target_probabilities=self._target_probabilities,
                                                        unique_targets=self._unique_targets)

    # noinspection PyProtectedMember,PyUnresolvedReferences
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:

        # get length of data, return random
        np.random.seed(123)
        # generate random `0` through `(len-1)` following the distribution found in data_y,
        # generate n=len(data_x) predictions
        random_predictions = np.random.choice(a=model_object._unique_targets,
                                              p=model_object._target_probabilities,
                                              size=len(data_x))
        # pd.Series(random_predictions).value_counts(normalize=True)

        # the generated numbers should correspond with the indexes of `_unique_targets`
        # for classification problems, _predict should return probabilities, so we will return 0/1's for each
        # class
        probabilities = pd.DataFrame()
        for target in model_object._unique_targets:
            probabilities[target] = pd.Series(data=[1 if x == target else 0 for x in random_predictions])

        probabilities.index = data_x.index  # ensure the index is the same
        return probabilities
