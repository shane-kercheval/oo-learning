import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictClassifierMixin, \
    SklearnPredictRegressorMixin


##############################################################################################################
# Classification
##############################################################################################################
class SvmLinearClassifierHP(HyperParamsBase):
    def __init__(self, penalty: str='l2', penalty_c: float=1.0, loss='hinge'):
        """
        for more info, see http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        :param penalty: "Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in
            SVC. The ‘l1’ leads to coef_ vectors that are sparse."
        :param penalty_c: "Penalty parameter C of the error term."
        :param loss: "Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC
            class) while ‘squared_hinge’ is the square of the hinge loss."
        """
        super().__init__()
        self._params_dict = dict(penalty=penalty, penalty_c=penalty_c, loss=loss)


class SvmLinearClassifier(SklearnPredictClassifierMixin, ModelWrapperBase):
    def __init__(self, fit_intercept=False):
        """
        :param fit_intercept: set to False by default, since the expectation is that One-Hot encoding will
            be used
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

        svm = LinearSVC(fit_intercept=self._fit_intercept,
                        penalty=param_dict['penalty'],
                        C=param_dict['penalty_c'],
                        loss=param_dict['loss'],
                        # Prefer dual=False when n_samples > n_features
                        # n_samples > n_features == data_x.shape[0] > data_x.shape[1]
                        # dual=False if data_x.shape[0] > data_x.shape[1] else True,
                        # but
                        # ValueError: Unsupported set of arguments: The combination of penalty=l2 and
                        # loss=hinge are not supported when dual=False, Parameters: penalty=l2, loss=hinge,
                        # dual=False
                        random_state=42)
        # SVM classifiers do not output probabilities for each class (but we will convert them to remain
        #
        # https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
        model_object = CalibratedClassifierCV(svm)
        model_object.fit(X=data_x, y=data_y)

        return model_object


class SvmPolynomialClassifierHP(HyperParamsBase):
    def __init__(self, degree: int=3, coef0: float=0.0, penalty_c: float=1.0):
        """
        for more info, see http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        :param degree: "Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels."
        :param coef0: "Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’."
        :param penalty_c: "Penalty parameter C of the error term."
        """
        super().__init__()
        self._params_dict = dict(degree=degree, coef0=coef0, penalty_c=penalty_c)


class SvmPolynomialClassifier(SklearnPredictClassifierMixin, ModelWrapperBase):
    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is not None
        param_dict = hyper_params.params_dict

        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        # noinspection SpellCheckingInspection
        svm = SVC(kernel='poly',
                  degree=param_dict['degree'],
                  coef0=param_dict['coef0'],
                  C=param_dict['penalty_c'],
                  random_state=42)
        # SVM classifiers do not output probabilities for each class (but we will convert them to remain
        # https://stackoverflow.com/questions/26478000/converting-linearsvcs-decision-function-to-probabilities-scikit-learn-python
        model_object = CalibratedClassifierCV(svm)
        model_object.fit(X=data_x, y=data_y)

        return model_object


##############################################################################################################
# Regression
##############################################################################################################
class SvmLinearRegressorHP(HyperParamsBase):
    # noinspection SpellCheckingInspection
    def __init__(self, epsilon: float=0.1, penalty_c: float=1.0):
        """
        for more info, see http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
        :param epsilon: "Epsilon parameter in the epsilon-insensitive loss function. Note that the value of
            this parameter depends on the scale of the target variable y. If unsure, set epsilon=0."
        :param penalty_c: "Penalty parameter C of the error term. The penalty is a squared l2 penalty. The
            bigger this parameter, the less regularization is used."
        """
        super().__init__()
        self._params_dict = dict(epsilon=epsilon, penalty_c=penalty_c)


class SvmLinearRegressor(SklearnPredictRegressorMixin, ModelWrapperBase):
    def __init__(self, fit_intercept=False):
        """
        :param fit_intercept: set to False by default, since the expectation is that One-Hot encoding will
            be used
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

        svm = LinearSVR(fit_intercept=self._fit_intercept,
                        C=param_dict['penalty_c'],
                        epsilon=param_dict['epsilon'],
                        random_state=42)
        svm.fit(X=data_x, y=data_y)
        return svm


class SvmPolynomialRegressorHP(HyperParamsBase):
    # noinspection SpellCheckingInspection
    def __init__(self, degree: int=3, epsilon: float=0.1, penalty_c: float=1.0):
        """
        for more info, see http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        :param degree: "Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels."
        :param epsilon: "Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no
            penalty is associated in the training loss function with points predicted within a distance
            epsilon from the actual value."
        :param penalty_c: "Penalty parameter C of the error term."
        """
        super().__init__()
        self._params_dict = dict(degree=degree, epsilon=epsilon, penalty_c=penalty_c)


class SvmPolynomialRegressor(SklearnPredictRegressorMixin, ModelWrapperBase):
    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is not None
        param_dict = hyper_params.params_dict

        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        svm = SVR(kernel='poly',
                  degree=param_dict['degree'],
                  C=param_dict['penalty_c'],
                  epsilon=param_dict['epsilon'])
        svm.fit(X=data_x, y=data_y)
        return svm
