import numpy as np
import pandas as pd
import sklearn.dummy

from oolearning.enums.DummyClassifierStrategy import DummyClassifierStrategy
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictClassifierMixin


class DummyClassifier(SklearnPredictClassifierMixin, ModelWrapperBase):
    def __init__(self, strategy: DummyClassifierStrategy):
        """
        Uses sklearn's DummyClassifier 
        :param strategy: how to 

        https://www.google.com/search?q=sklearn+dummy+classifier&rlz=1C5CHFA_enUS693US694&oq=Sklearn+DUmmy+Class&aqs=chrome.0.0j69i57j0.4456j0j1&sourceid=chrome&ie=UTF-8
        """
        super().__init__()
        self._strategy = strategy

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        assert hyper_params is None

        np.random.seed(42)
        model_object = sklearn.dummy.DummyClassifier(strategy=self._strategy.value, random_state=42)
        model_object.fit(X=data_x, y=data_y)

        return model_object
