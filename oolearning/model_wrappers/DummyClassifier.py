import numpy as np
import pandas as pd
import sklearn.dummy
from matplotlib import figure

from oolearning.enums.DummyClassifierStrategy import DummyClassifierStrategy
from oolearning.model_wrappers.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class DummyClassifierFI(FittedInfoBase):

    @property
    def results_summary(self) -> object:
        return None

    @property
    def feature_importance(self) -> dict:
        # noinspection PyTypeChecker
        return None

    @property
    def graph(self) -> figure.Figure:
        # noinspection PyTypeChecker
        return None


class DummyClassifier(ModelWrapperBase):
    def __init__(self, strategy: DummyClassifierStrategy):
        """
        need to set fit_intercept to False if using One-Hot-Encoding
        strategy:
        """
        super().__init__()
        self._strategy = strategy

    def _create_fitted_info_object(self,
                                   model_object, data_x: pd.DataFrame,
                                   data_y: np.ndarray,
                                   hyper_params: HyperParamsBase = None) -> FittedInfoBase:
        return DummyClassifierFI(model_object=model_object,
                                 feature_names=data_x.columns.values.tolist(),
                                 hyper_params=None)  # Regression does not have any hyper-parameters

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        assert hyper_params is None

        np.random.seed(42)
        model_object = sklearn.dummy.DummyClassifier(strategy=self._strategy.value, random_state=42)
        model_object.fit(X=data_x, y=data_y)

        return model_object

    # noinspection PyUnresolvedReferences
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(model_object.predict_proba(data_x))
        predictions.columns = model_object.classes_

        return predictions
