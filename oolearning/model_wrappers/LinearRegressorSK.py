import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictArrayMixin


class LinearRegressorSK(SklearnPredictArrayMixin, ModelWrapperBase):
    """
    fits Linear Regression model on the data

    Uses sklearn's LinearRegression object rather than `statsmodels` like the `LinearRegressor` object
        (the `statsmodels` implementation freezes with parallelization).
    """
    def __init__(self, fit_intercept=True):
        super().__init__()
        self._fit_intercept = fit_intercept

    @property
    def feature_importance(self):
        return None

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is None

        # Regression can't handle missing values
        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        if any(np.isnan(data_y)):
            raise MissingValueError()

        model = LinearRegression(fit_intercept=self._fit_intercept)
        model.fit(data_x, data_y)
        return model
