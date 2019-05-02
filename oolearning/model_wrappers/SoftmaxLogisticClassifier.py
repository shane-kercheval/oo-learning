import numpy as np
import pandas as pd
from sklearn import linear_model

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictProbabilityMixin


class SoftmaxLogisticHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticClassifier.html for more
    information
    on tuning parameters
    """
    # noinspection SpellCheckingInspection
    def __init__(self, regularization_inverse: float = 1.0, solver: str = 'lbfgs'):
        # http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        # (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’ and ‘newton-cg’
        # solvers.)
        # "The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties."
        # it would be misleading to give the option for `penalty` since it can only be l2, so it is manually
        # set in the call to sklearn.LogisticRegression
        super().__init__()

        assert solver == 'newton-cg' or solver == 'sag' or solver == 'lbfgs'
        self._params_dict = dict(regularization_inverse=regularization_inverse,
                                 solver=solver)


class SoftmaxLogisticClassifier(SklearnPredictProbabilityMixin, ModelWrapperBase):
    def __init__(self, fit_intercept: bool = True, seed: int = 42):
        """
        need to set fit_intercept to False if using One-Hot-Encoding
        :param fit_intercept:
        """
        super().__init__()
        self._fit_intercept = fit_intercept
        self._seed = seed

    @property
    def feature_importance(self):
        return None

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: SoftmaxLogisticHP = None) -> object:

        assert hyper_params is not None
        assert isinstance(hyper_params, SoftmaxLogisticHP)
        param_dict = hyper_params.params_dict

        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        # noinspection SpellCheckingInspection
        model_object = linear_model.LogisticRegression(multi_class='multinomial',
                                                       fit_intercept=self._fit_intercept,
                                                       penalty='l2',
                                                       C=param_dict['regularization_inverse'],
                                                       solver=param_dict['solver'],
                                                       random_state=self._seed)
        model_object.fit(X=data_x, y=data_y)

        return model_object
