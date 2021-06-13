from enum import unique, Enum
from typing import Union

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictArrayMixin, \
    SklearnPredictProbabilityMixin


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


# noinspection SpellCheckingInspection
@unique
class XGBEvalMetric(Enum):
    """
    Not all options are included. See http://xgboost.readthedocs.io/en/latest/parameter.html for all options

    “rmse”: root mean square error
    “mae”: mean absolute error
    “logloss”: negative log-likelihood
    “error”: Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the
        predictions, the evaluation will regard the instances with prediction value larger than 0.5 as
        positive instances, and the others as negative instances.
    “error@t”: a different than 0.5 binary classification threshold value could be specified by providing a
        numerical value through ‘t’.
    “merror”: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
    “mlogloss”: Multiclass logloss
    “auc”: Area under the curve for ranking evaluation.
    “ndcg”:Normalized Discounted Cumulative Gain
    “map”:Mean average precision
    “ndcg@n”,”map@n”: n can be assigned as an integer to cut off the top positions in the lists for evaluation.
    “ndcg-”,”map-”,”ndcg@n-”,”map@n-”: In XGBoost, NDCG and MAP will evaluate the score of a list without any
        positive samples as 1. By adding “-” in the evaluation metric XGBoost will evaluate these score as 0
        to be consistent under some conditions. training repeatedly
    """
    RMSE = 'rmse'
    MAE = 'mae'
    LOGLOSS = 'logloss'
    ERROR = 'error'
    MERROR = 'merror'
    MLOGLOSS = 'mlogloss'
    AUC = 'auc'
    NDCG = 'ndcg'
    MAP = 'map'


class XGBoostLinearHP(HyperParamsBase):
    # noinspection SpellCheckingInspection
    def __init__(self,
                 objective: XGBObjective,
                 n_estimators: int = 100,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 0.0):
        """
        See
            https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
            for information on Hyper-Parameters.

        Note: according to
            https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
            gblinear ony has n_estimators, alpha, and lambda to tune.

        :param objective:
        :param n_estimators:
        :param reg_alpha:
        :param reg_lambda:
        """
        super().__init__()

        # noinspection SpellCheckingInspection
        self._params_dict = dict(max_depth=3,
                                 learning_rate=0.1,
                                 n_estimators=n_estimators,
                                 verbosity=0,
                                 objective=objective.value,
                                 booster='gblinear',
                                 n_jobs=1,
                                 nthread=1,
                                 gamma=0.0,
                                 min_child_weight=1,
                                 max_delta_step=0,
                                 subsample=1.0,
                                 colsample_bytree=1.0,
                                 colsample_bylevel=1.0,
                                 reg_alpha=reg_alpha,
                                 reg_lambda=reg_lambda,
                                 scale_pos_weight=1.0,
                                 base_score=0.5,
                                 missing=np.nan)


# noinspection SpellCheckingInspection
class XGBoostTreeHP(HyperParamsBase):
    def __init__(self,
                 objective: XGBObjective,
                 max_depth: int = 3,
                 learning_rate: float = 0.1,
                 n_estimators: int = 100,
                 verbosity: int = 0,
                 n_jobs: int = 1,
                 nthread: int = 1,
                 gamma: float = 0.0,
                 min_child_weight: int = 1,
                 max_delta_step: int = 0,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 colsample_bylevel: float = 1.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 scale_pos_weight: float = 1.0,
                 base_score: float = 0.5,
                 missing: float = np.nan):
        """
        See
            https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/
            for information on Hyper-Parameters.
        """
        super().__init__()

        self._params_dict = dict(max_depth=max_depth,
                                 learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 verbosity=verbosity,
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
                                 missing=missing)


# TODO: not implemented
class XGBoostDartHP(HyperParamsBase):
    def __init__(self):
        super().__init__()

        # noinspection SpellCheckingInspection
        self._params_dict = dict(booster='dart')


# noinspection SpellCheckingInspection,PyAbstractClass
class XGBoostBase(ModelWrapperBase):
    def __init__(self,
                 early_stopping_rounds: int = None,
                 eval_metric: Union[XGBEvalMetric, str, callable, None] = None,
                 eval_set: Union[list, None] = None,
                 seed: int = 42):
        """
        Note: sklearn takes an additional parameter (in the fit method) called eval_set. I simply set the
            eval set to the training set based to `train()`.

        param descriptions from: https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py

        :param eval_metric:
            If a str, should be a built-in evaluation metric to use. See
            doc/parameter.md. If callable, a custom evaluation metric. The call
            signature is func(y_predicted, y_true) where y_true will be a
            DMatrix object such that you may need to call the get_label
            method. It must return a str, value pair where the str is a name
            for the evaluation and value is the value of the evaluation
            function. This objective is always minimized.
        :param early_stopping_rounds:
            Activates early stopping. Validation error needs to decrease at
            least every <early_stopping_rounds> round(s) to continue training.
            Requires at least one item in evals.  If there's more than one,
            will use the last. Returns the model from the last iteration
            (not the best one). If early stopping occurs, the model will
            have three additional fields: bst.best_score, bst.best_iteration
            and bst.best_ntree_limit.
            (Use bst.best_ntree_limit to get the correct value if num_parallel_tree
            and/or num_class appears in the parameters)
        :param eval_set:
            A list of (X, y) tuple pairs to use as a validation set for
            early-stopping

            If eval_set is not supplied, but the other early stopping parameters are, then the `data_x` &
                `data_y` datasets that are passed to `train()` are used.
        """
        super().__init__()

        # must pass both parameters or neither
        assert (early_stopping_rounds is None and eval_metric is None) or \
               (early_stopping_rounds is not None and eval_metric is not None)

        self._early_stopping_rounds = early_stopping_rounds
        self._eval_metric = eval_metric.value if isinstance(eval_metric, XGBEvalMetric) else eval_metric
        self._eval_set = eval_set

        self._data_x_columns = None
        self._seed = seed

    @property
    def feature_importance(self):
        return None

    def plot_feature_importance(self):
        return None

    def xgb_fit(self, model, data_x, data_y):
        self._data_x_columns = data_x.columns.values

        if self._eval_metric is not None and self._eval_set is None:  # if early stopping but not eval set
            self._eval_set = [(data_x, data_y)]
        model.fit(X=data_x,
                  y=data_y,
                  early_stopping_rounds=self._early_stopping_rounds,
                  eval_metric=self._eval_metric,
                  eval_set=self._eval_set)

        return model


# noinspection SpellCheckingInspection
class XGBoostRegressor(SklearnPredictArrayMixin, XGBoostBase):
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
                             verbosity=param_dict['verbosity'],
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
                             random_state=self._seed,
                             # seed=param_dict['seed'],
                             missing=param_dict['missing'])

        return self.xgb_fit(model=model, data_x=data_x, data_y=data_y)


# noinspection SpellCheckingInspection
class XGBoostClassifier(SklearnPredictProbabilityMixin, XGBoostBase):

    @property
    def feature_importance(self):
        return pd.DataFrame.from_dict(self.model_object.get_booster().get_score(importance_type='gain'),
                                      orient='index',
                                      columns=['gain_values']).sort_values('gain_values', ascending=False)

    def plot_feature_importance(self):
        return plot_importance(self.model_object, importance_type='gain', max_num_features=30)

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
                              verbosity=param_dict['verbosity'],
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
                              random_state=self._seed,
                              # seed=param_dict['seed'],
                              # use_label_encoder=False,
                              missing=param_dict['missing'])

        return self.xgb_fit(model=model, data_x=data_x, data_y=data_y)
