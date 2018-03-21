import numpy as np
import pandas as pd
from matplotlib import figure
from sklearn.linear_model import Lasso

from oolearning.model_wrappers.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class LassoRegressionHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html for more information
    on tuning parameters
    """

    # noinspection SpellCheckingInspection
    def __init__(self, alpha: float=0.5):
        super().__init__()
        self._params_dict = dict(alpha=alpha)


class LassoRegressionFI(FittedInfoBase):
    @property
    def results_summary(self) -> object:
        raise NotImplementedError()

    @property
    def feature_importance(self) -> dict:
        raise NotImplementedError()

    @property
    def graph(self) -> figure.Figure:
        raise NotImplementedError()


class LassoRegression(ModelWrapperBase):
    """
    fits Linear Regression model on the data
    """

    def __init__(self, fit_intercept=True):
        super().__init__()
        self._fit_intercept = fit_intercept

    def _create_fitted_info_object(self,
                                   model_object,
                                   data_x: pd.DataFrame,
                                   data_y: np.ndarray,
                                   hyper_params: HyperParamsBase=None) -> FittedInfoBase:
        return LassoRegressionFI(model_object=model_object,
                                 feature_names=data_x.columns.values.tolist(),
                                 hyper_params=hyper_params)

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
        ridge_reg = Lasso(alpha=param_dict['alpha'],
                          fit_intercept=True,
                          random_state=42)
        ridge_reg.fit(data_x, data_y)
        return ridge_reg

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return model_object.predict(data_x)
