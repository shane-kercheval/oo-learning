import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from oolearning.fitted_info.LogisticFI import LogisticFI
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.fitted_info.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class LogisticMW(ModelWrapperBase):

    def _create_fitted_info_object(self, model_object, data_x: pd.DataFrame, data_y: np.ndarray,
                                   hyper_params: HyperParamsBase = None) -> FittedInfoBase:
        return LogisticFI(model_object=model_object,
                          feature_names=data_x.columns.values.tolist(),
                          hyper_params=None)  # Regression does not have any hyper-parameters

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        assert hyper_params is None

        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        model_object = LogisticRegression(fit_intercept=True, C=1e9, n_jobs=-1, random_state=42)
        model_object.fit(X=data_x, y=data_y)

        return model_object

    # noinspection PyUnresolvedReferences
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(model_object.predict_proba(data_x))
        predictions.columns = model_object.classes_

        return predictions
