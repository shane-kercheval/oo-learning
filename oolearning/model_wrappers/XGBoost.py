from typing import Union

import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier

from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictRegressorMixin, \
    SklearnPredictClassifierMixin
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase

from enum import unique, Enum


# noinspection SpellCheckingInspection
@unique
class XGBObjective(Enum):
    """
    http://xgboost-clone.readthedocs.io/en/latest/parameter.html#learning-task-parameters

    objective [ default=reg:linear ]
    specify the learning task and the corresponding learning objective, and the objective options are below:
    “reg:linear” –linear regression
    “reg:logistic” –logistic regression
    “binary:logistic” –logistic regression for binary classification, output probability
    “binary:logitraw” –logistic regression for binary classification, output score before logistic
        transformation
    “count:poisson” –poisson regression for count data, output mean of poisson distribution
    max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
    “multi:softmax” –set XGBoost to do multiclass classification using the softmax objective, you also need to
        set num_class(number of classes)
    “multi:softprob” –same as softmax, but output a vector of ndata * nclass, which can be further reshaped to
        ndata, nclass matrix. The result contains predicted probability of each data point belonging to each
        class.
    “rank:pairwise” –set XGBoost to do ranking task by minimizing the pairwise loss
    """
    REG_LINEAR = 'reg:linear'
    REG_LOGISTIC = 'reg:logistic'
    BINARY_LOGISTIC = 'binary:logistic'
    BINARY_LOGITRAW = 'binary:logitraw'
    COUNT_POISSON = 'count:poisson'
    MULTI_SOFTMAX = 'multi:softmax'
    MULTI_SOFTPROB = 'multi:softprob'
    RANK_PAIRWISE = 'rank:pairwise'


class XGBoostLinearHP(HyperParamsBase):
    def __init__(self,
                 objective: XGBObjective,
                 n_estimators: int=100,
                 alpha: float=0,
                 lambda_r: float=1):
        """
        See
            https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
            for information on Hyper-Parameters.

        Note: according to
            https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
            gblinear ony has n_estimators, alpha, and lambda to tune.

        :param objective:
        :param n_estimators:
        :param alpha:
        :param lambda_r:
        """
        super().__init__()

        # noinspection SpellCheckingInspection
        self._params_dict = dict(max_depth=3,
                                 learning_rate=0.1,
                                 n_estimators=n_estimators,
                                 silent=True,
                                 objective=objective.value,
                                 booster='gblinear',
                                 n_jobs=1,
                                 nthread=None,
                                 gamma=0,
                                 min_child_weight=1,
                                 max_delta_step=0,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 reg_alpha=alpha,
                                 reg_lambda=lambda_r,
                                 scale_pos_weight=1,
                                 base_score=0.5,
                                 random_state=42,
                                 seed=None,
                                 missing=None)


# noinspection SpellCheckingInspection
class XGBoostTreeHP(HyperParamsBase):
    def __init__(self,
                 objective: XGBObjective,
                 max_depth: int=3,
                 learning_rate: float=0.1,
                 n_estimators: int=100,
                 silent: bool=True,
                 n_jobs: int=1,
                 nthread: int=None,
                 gamma: float=0,
                 min_child_weight: int=1,
                 max_delta_step: int=0,
                 subsample: float=1,
                 colsample_bytree: float=1,
                 colsample_bylevel: float=1,
                 reg_alpha: float=0,
                 reg_lambda: float=1,
                 scale_pos_weight: float=1,
                 base_score: float=0.5,
                 random_state: int=42,
                 seed: int=None,
                 missing: float=None):
        """
        See
            https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
            for information on Hyper-Parameters.
        """
        super().__init__()

        self._params_dict = dict(max_depth=max_depth,
                                 learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 silent=silent,
                                 objective=objective.value,
                                 booster='gbtree',
                                 n_jobs=n_jobs,
                                 nthread=nthread,
                                 gamma=gamma,
                                 min_child_weight=min_child_weight,
                                 max_delta_step=max_delta_step,
                                 subsample=subsample,
                                 colsample_bytree=colsample_bytree,
                                 colsample_bylevel=colsample_bylevel,
                                 reg_alpha=reg_alpha,
                                 reg_lambda=reg_lambda,
                                 scale_pos_weight=scale_pos_weight,
                                 base_score=base_score,
                                 random_state=random_state,
                                 seed=seed,
                                 missing=missing)


# TODO: not implemented
class XGBoostDartHP(HyperParamsBase):
    def __init__(self):
        super().__init__()

        # noinspection SpellCheckingInspection
        self._params_dict = dict(booster='dart')


# noinspection SpellCheckingInspection
class XGBoostRegressor(SklearnPredictRegressorMixin, ModelWrapperBase):

    @property
    def feature_importance(self):
        pass

    # noinspection PyTypeChecker
    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is not None
        assert isinstance(hyper_params, XGBoostLinearHP) or isinstance(hyper_params, XGBoostTreeHP)

        param_dict = hyper_params.params_dict

        model = XGBRegressor(max_depth=param_dict['max_depth'],
                             learning_rate=param_dict['learning_rate'],
                             n_estimators=param_dict['n_estimators'],
                             silent=param_dict['silent'],
                             objective=param_dict['objective'],
                             booster=param_dict['booster'],
                             n_jobs=param_dict['n_jobs'],
                             nthread=param_dict['nthread'],
                             gamma=param_dict['gamma'],
                             min_child_weight=param_dict['min_child_weight'],
                             max_delta_step=param_dict['max_delta_step'],
                             subsample=param_dict['subsample'],
                             colsample_bytree=param_dict['colsample_bytree'],
                             colsample_bylevel=param_dict['colsample_bylevel'],
                             reg_alpha=param_dict['reg_alpha'],
                             reg_lambda=param_dict['reg_lambda'],
                             scale_pos_weight=param_dict['scale_pos_weight'],
                             base_score=param_dict['base_score'],
                             random_state=param_dict['random_state'],
                             seed=param_dict['seed'],
                             missing=param_dict['missing'])

        model.fit(X=data_x, y=data_y)

        return model


# noinspection SpellCheckingInspection
class XGBoostClassifier(SklearnPredictClassifierMixin, ModelWrapperBase):

    @property
    def feature_importance(self):
        pass

    # noinspection PyTypeChecker
    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is not None
        assert isinstance(hyper_params, XGBoostTreeHP)

        param_dict = hyper_params.params_dict

        model = XGBClassifier(max_depth=param_dict['max_depth'],
                              learning_rate=param_dict['learning_rate'],
                              n_estimators=param_dict['n_estimators'],
                              silent=param_dict['silent'],
                              objective=param_dict['objective'],
                              booster=param_dict['booster'],
                              n_jobs=param_dict['n_jobs'],
                              nthread=param_dict['nthread'],
                              gamma=param_dict['gamma'],
                              min_child_weight=param_dict['min_child_weight'],
                              max_delta_step=param_dict['max_delta_step'],
                              subsample=param_dict['subsample'],
                              colsample_bytree=param_dict['colsample_bytree'],
                              colsample_bylevel=param_dict['colsample_bylevel'],
                              reg_alpha=param_dict['reg_alpha'],
                              reg_lambda=param_dict['reg_lambda'],
                              scale_pos_weight=param_dict['scale_pos_weight'],
                              base_score=param_dict['base_score'],
                              random_state=param_dict['random_state'],
                              seed=param_dict['seed'],
                              missing=param_dict['missing'])

        model.fit(X=data_x, y=data_y)

        return model
