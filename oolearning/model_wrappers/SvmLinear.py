import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictClassifierMixin


class SvmLinearHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticClassifier.html for more
    information
    on tuning parameters
    """

    # noinspection SpellCheckingInspection
    def __init__(self, penalty: str='l2', penalty_c: float=1.0, loss='hinge'):
        # "The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties."
        super().__init__()
        self._params_dict = dict(penalty=penalty, penalty_c=penalty_c, loss=loss)


class SvmLinearClassifier(SklearnPredictClassifierMixin, ModelWrapperBase):
    def __init__(self, fit_intercept=True):
        """
        need to set fit_intercept to False if using One-Hot-Encoding
        :param fit_intercept:
        """
        super().__init__()
        self._fit_intercept = fit_intercept

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is not None
        param_dict = hyper_params.params_dict

        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        np.random.seed(42)
        # noinspection SpellCheckingInspection
        svm = LinearSVC(fit_intercept=self._fit_intercept,
                        penalty=param_dict['penalty'],
                        C=param_dict['penalty_c'],
                        loss=param_dict['loss'],
                        random_state=42)
        # https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
        model_object = CalibratedClassifierCV(svm)
        model_object.fit(X=data_x, y=data_y)

        return model_object
