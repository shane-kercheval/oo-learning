import numpy as np
import pandas as pd
import sklearn.dummy

from oolearning.enums.DummyClassifierStrategy import DummyClassifierStrategy
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictProbabilityMixin


class DummyClassifier(SklearnPredictProbabilityMixin, ModelWrapperBase):
    def __init__(self, strategy: DummyClassifierStrategy):
        """
        Uses sklearn's DummyClassifier 

        Supported strategies:

        ```
            “stratified”: generates predictions by respecting the training set’s class distribution.
            “most_frequent”: always predicts the most frequent label in the training set.
            “prior”: always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior.
            “uniform”: generates predictions uniformly at random.
        ```

        - http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

        :param strategy: A DummyClassifierStrategy enum. 
        """
        super().__init__()
        self._strategy = strategy

    @property
    def feature_importance(self):
        return None

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        assert hyper_params is None

        np.random.seed(42)
        model_object = sklearn.dummy.DummyClassifier(strategy=self._strategy.value, random_state=42)
        model_object.fit(X=data_x, y=data_y)

        return model_object
