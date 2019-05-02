import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictArrayMixin


class LassoRegressorHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html for more information
    on tuning parameters
    """

    # noinspection SpellCheckingInspection
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self._params_dict = dict(alpha=alpha)


class LassoRegressor(SklearnPredictArrayMixin, ModelWrapperBase):
    """
    fits Linear Regression model on the data
    """

    def __init__(self, fit_intercept: bool = True, seed: int = 42):
        super().__init__()
        self._fit_intercept = fit_intercept
        self._seed = seed

    @property
    def feature_importance(self):
        return None

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: LassoRegressorHP = None) -> object:

        assert hyper_params is not None
        assert isinstance(hyper_params, LassoRegressorHP)
        param_dict = hyper_params.params_dict

        # Regression can't handle missing values
        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        if any(np.isnan(data_y)):
            raise MissingValueError()
        ridge_reg = Lasso(alpha=param_dict['alpha'],
                          fit_intercept=True,
                          random_state=self._seed)
        ridge_reg.fit(data_x, data_y)
        return ridge_reg
