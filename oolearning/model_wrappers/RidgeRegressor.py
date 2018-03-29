import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictRegressorMixin


class RidgeRegressorHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html for more information
    on tuning parameters
    """

    # noinspection SpellCheckingInspection
    def __init__(self, alpha: float=0.5, solver: str='cholesky'):
        super().__init__()
        self._params_dict = dict(alpha=alpha, solver=solver)


class RidgeRegressor(SklearnPredictRegressorMixin, ModelWrapperBase):
    """
    fits Linear Regression model on the data
    """

    def __init__(self, fit_intercept=True):
        super().__init__()
        self._fit_intercept = fit_intercept

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase=None) -> object:

        assert hyper_params is not None
        param_dict = hyper_params.params_dict

        # Regression can't handle missing values
        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        if any(np.isnan(data_y)):
            raise MissingValueError()

        ridge_reg = Ridge(alpha=param_dict['alpha'],
                          solver=param_dict['solver'],
                          fit_intercept=self._fit_intercept,
                          random_state=42)
        ridge_reg.fit(data_x, data_y)
        return ridge_reg
