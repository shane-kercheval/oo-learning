import pandas as pd
import numpy as np
import statsmodels.api as sm

from oolearning.fitted_info.RegressionFI import RegressionFI
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.fitted_info.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class RegressionMW(ModelWrapperBase):
    """
    fits Linear Regression model on the data
    """

    def __init__(self):
        super().__init__()

    def _create_fitted_info_object(self,
                                   model_object,
                                   data_x: pd.DataFrame,
                                   data_y: np.ndarray,
                                   hyper_params: HyperParamsBase=None) -> FittedInfoBase:
        return RegressionFI(model_object=model_object,
                            feature_names=data_x.columns.values.tolist(),
                            hyper_params=None,  # Regression does not have any hyper-parameters
                            training_target_std=np.std(data_y))

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase=None) -> object:

        assert hyper_params is None  # no hyper-params for regression

        # Regression can't handle missing values
        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        if any(np.isnan(data_y)):
            raise MissingValueError()

        model_object = sm.OLS(data_y, sm.add_constant(data_x)).fit()
        return model_object

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return model_object.predict(sm.add_constant(data_x))
