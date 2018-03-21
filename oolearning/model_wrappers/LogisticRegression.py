import numpy as np
import pandas as pd
from matplotlib import figure
from sklearn import linear_model

from oolearning.model_wrappers.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


# noinspection PyPropertyDefinition
class LogisticRegressionFI(FittedInfoBase):
    @property
    def results_summary(self) -> pd.DataFrame:
        #return self._model_object.summary()
        pass

    @property
    def feature_importance(self) -> dict:
        pass

    @property
    def graph(self) -> figure.Figure:
        pass


class LogisticRegressionHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for more
    information
    on tuning parameters
    """

    # noinspection SpellCheckingInspection
    def __init__(self, penalty: str='l2', regularization_inverse: float=1.0):
        super().__init__()
        self._params_dict = dict(penalty=penalty, regularization_inverse=regularization_inverse)


class LogisticRegression(ModelWrapperBase):
    def __init__(self, fit_intercept=True):
        """
        need to set fit_intercept to False if using One-Hot-Encoding
        :param fit_intercept:
        """
        super().__init__()
        self._fit_intercept = fit_intercept

    def _create_fitted_info_object(self, model_object, data_x: pd.DataFrame, data_y: np.ndarray,
                                   hyper_params: HyperParamsBase = None) -> FittedInfoBase:
        return LogisticRegressionFI(model_object=model_object,
                                    feature_names=data_x.columns.values.tolist(),
                                    hyper_params=hyper_params)

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        assert hyper_params is not None
        param_dict = hyper_params.params_dict

        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        np.random.seed(42)
        model_object = linear_model.LogisticRegression(fit_intercept=self._fit_intercept,
                                                       penalty=param_dict['penalty'],
                                                       C=param_dict['regularization_inverse'],
                                                       random_state=42)
        model_object.fit(X=data_x, y=data_y)

        return model_object

    # noinspection PyUnresolvedReferences
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(model_object.predict_proba(data_x))
        predictions.columns = model_object.classes_

        return predictions
