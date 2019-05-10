from enum import unique, Enum
from typing import Union

import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import plot_importance

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictArrayMixin, \
    SklearnPredictProbabilityMixin


# noinspection SpellCheckingInspection
# @unique
# class LightGBMEvalMetric(Enum):
#     """
#     https://lightgbm.readthedocs.io/en/latest/Parameters.html
#     `l1`, absolute loss, aliases: mean_absolute_error, mae, regression_l1
#     `l2`, square loss, aliases: mean_squared_error, mse, regression_l2, regression
#     `l2_root`, root square loss, aliases: root_mean_squared_error, rmse
#     `quantile`, Quantile regression
#     `mape`, MAPE loss, aliases: mean_absolute_percentage_error
#     `huber`, Huber loss
#     `fair`, Fair loss
#     `poisson`, negative log-likelihood for Poisson regression
#     `gamma`, negative log-likelihood for Gamma regression
#     `gamma_deviance`, residual deviance for Gamma regression
#     `tweedie`, negative log-likelihood for Tweedie regression
#     `ndcg`, NDCG, aliases: lambdarank
#     `map`, MAP, aliases: mean_average_precision
#     `auc`, AUC
#     `binary_logloss`, log loss, aliases: binary
#     `binary_error`, for one sample: 0 for correct classification, 1 for error classification
#     `multi_logloss`, log loss for multi-class classification, aliases: multiclass, softmax, multiclassova, multiclass_ova, ova, ovr
#     `multi_error`, error rate for multi-class classification
#     `xentropy`, cross-entropy (with optional linear weights), aliases: cross_entropy
#     `xentlambda`, “intensity-weighted” cross-entropy, aliases: cross_entropy_lambda
#     `kldiv`, Kullback-Leibler divergence, aliases: kullback_leibler
#     """
#     L1 = 'l1'
#     L2 = 'l2'
#     L2_ROOT = 'l2_root'
#     QUANTILE = 'quantile'
#     MAPE = 'mape'
#     HUBER = 'huber'
#     FAIR = 'fair'
#     POISSON = 'poisson'
#     GAMMA = 'gamma'
#     GAMMA_DEVIANCE = 'gamma_deviance'
#     TWEEDIE = 'tweedie'
#     NDCG = 'ndcg'
#     MAP = 'map'
#     AUC = 'auc'
#     BINARY_LOGLOSS = 'binary_logloss'
#     BINARY_ERROR = 'binary_error'
#     MULTI_LOGLOSS = 'multi_logloss'
#     MULTI_ERROR = 'multi_error'
#     XENTROPY = 'xentropy'
#     XENTLAMBDA = 'xentlambda'
#     KLDIV = 'kldiv'

@unique
class LightGBMBoostingType(Enum):
    GRADIENT_BOOSTING_DECISION_TREE = 'gbdt'
    RANDOM_FOREST = 'random_forest'
    DROPOUTS_MEET_MULTIPLE_ADDITIVE_REGRESSION_TREES = 'dart'
    GRADIENT_BASED_ONE_SIDE_SAMPLING = 'goss'


@unique
class LightGBMObjective(Enum):
    REGRESSION = 'regression'  # for LGBMRegressor
    BINARY = 'binary'  # for LGBMClassifier
    MULTI_CLASS = 'multiclass'  # for LGBMClassifier


class LightGBMHP(HyperParamsBase):
    """
    https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html

    For Faster Speed
        Use bagging by setting `bagging_fraction` and `bagging_freq`
        Use feature sub-sampling by setting `feature_fraction`
        Use small `max_bin`
        Use `save_binary` to speed up data loading in future learning
        Use parallel learning, refer to Parallel Learning Guide
    For Better Accuracy
        Use large `max_bin` (may be slower)
        Use small `learning_rate` with large `num_iterations`
        Use large `num_leaves` (may cause over-fitting)
        Use bigger training data
        Try `dart`
    Deal with Over-fitting
        Use small `max_bin`
        Use small `num_leaves`
        Use `min_data_in_leaf` and `min_sum_hessian_in_leaf`
        Use bagging by set `bagging_fraction` and `bagging_freq`
        Use feature sub-sampling by set `feature_fraction`
        Use bigger training data
        Try `lambda_l1`, `lambda_l2` and `min_gain_to_split` for regularization
        Try `max_depth` to avoid growing deep tree
    """

    def __init__(self,
                 # most important #######
                 num_leaves: int = 31,
                 min_data_in_leaf: int = 20,
                 max_depth: int = -1,
                 ########################
                 bagging_fraction: float = 1.0,
                 bagging_freq: int = 0,
                 feature_fraction: float = 1.0,
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 0.0,
                 learning_rate: float = 0.1,
                 max_bin: int = 255,
                 min_gain_to_split: float = 0.0,
                 min_sum_hessian_in_leaf: float = 1e-3,
                 n_estimators: int = 100,
                 # is_unbalanced: bool = False,
                 scale_pos_weight: float = 1.0,
                 # save_binary: bool = False,
                 match_type=False):

        """
        https://lightgbm.readthedocs.io/en/latest/Parameters.html

        :param num_leaves:
        :param min_data_in_leaf:
        :param max_depth:
        :param bagging_fraction:
        :param bagging_freq:
            frequency for bagging
            0 means disable bagging; k means perform bagging at every k iteration
            Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as well
        :param feature_fraction:
        :param lambda_l1:
        :param lambda_l2:
        :param learning_rate:
        :param max_bin:
        :param min_gain_to_split:
        :param min_sum_hessian_in_leaf:
        :param n_estimators:
            alias for num_iterations
        :param scale_pos_weight:
            used only in binary application
            weight of labels with positive class
            Note: this parameter cannot be used at the same time with is_unbalance, choose only one of them
        """

        assert num_leaves > 1
        assert max_depth >= -1
        assert 0.0 < bagging_fraction <= 1.0
        assert bagging_freq >= 0
        assert 0.0 < feature_fraction <= 1.0
        assert lambda_l1 >= 0.0
        assert lambda_l2 >= 0.0
        assert learning_rate > 0.0
        assert max_bin > 1
        assert min_gain_to_split >= 0.0
        assert min_sum_hessian_in_leaf >= 0.0
        assert n_estimators >= 0
        # assert isinstance(is_unbalanced, bool)
        assert scale_pos_weight > 0.0
        # assert isinstance(save_binary, bool)

        # TODO REMOVE
        # REQUIRED FOR BAYESIAN OPTIMIZATION
        num_leaves = int(round(num_leaves))
        min_data_in_leaf = int(round(min_data_in_leaf))
        max_depth = int(round(max_depth))
        bagging_freq = int(round(bagging_freq))
        max_bin = int(round(max_bin))
        n_estimators = int(round(n_estimators))

        super().__init__(match_type=match_type)

        self._params_dict = dict(num_leaves=num_leaves,
                                 min_data_in_leaf=min_data_in_leaf,
                                 max_depth=max_depth,
                                 bagging_fraction=bagging_fraction,
                                 bagging_freq=bagging_freq,
                                 feature_fraction=feature_fraction,
                                 lambda_l1=lambda_l1,
                                 lambda_l2=lambda_l2,
                                 learning_rate=learning_rate,
                                 max_bin=max_bin,
                                 min_gain_to_split=min_gain_to_split,
                                 min_sum_hessian_in_leaf=min_sum_hessian_in_leaf,
                                 n_estimators=n_estimators,
                                 # is_unbalanced=is_unbalanced,
                                 scale_pos_weight=scale_pos_weight)


# noinspection PyAbstractClass
class LightGBMBase(ModelWrapperBase):

    def __init__(self,
                 boosting_type: LightGBMBoostingType = LightGBMBoostingType.GRADIENT_BOOSTING_DECISION_TREE,
                 # early_stopping_rounds: int = None,
                 seed: int = 42):
        super().__init__()

        self._boosting_type = boosting_type.value
        # self._early_stopping_rounds = early_stopping_rounds
        self._seed = seed

    @property
    def feature_importance(self):
        return pd.DataFrame(self.model_object.booster_.feature_importance(importance_type='gain'),
                            index=self.model_object.booster_.feature_name(),
                            columns=['gain_values']).sort_values('gain_values', ascending=False)

    def plot_feature_importance(self):
        return plot_importance(self.model_object, importance_type='gain', max_num_features=30)


class LightGBMClassifier(SklearnPredictProbabilityMixin, LightGBMBase):

    def _train(self,
               data_x: pd.DataFrame,
               data_y: Union[np.ndarray, None],
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is not None

        param_dict = hyper_params.params_dict
        lgb_classifier = lgb.LGBMClassifier(boosting_type=self._boosting_type,
                                            num_leaves=param_dict['num_leaves'],
                                            max_depth=param_dict['max_depth'],
                                            learning_rate=param_dict['learning_rate'],
                                            n_estimators=param_dict['n_estimators'],
                                            subsample_for_bin=200000,
                                            objective=LightGBMObjective.BINARY.value,
                                            # is_unbalanced=param_dict['is_unbalanced'],
                                            scale_pos_weight=param_dict['scale_pos_weight'],
                                            min_split_gain=param_dict['min_gain_to_split'],
                                            min_child_weight=param_dict['min_sum_hessian_in_leaf'],
                                            min_child_samples=param_dict['min_data_in_leaf'],
                                            subsample=param_dict['bagging_fraction'],
                                            subsample_freq=param_dict['bagging_freq'],
                                            colsample_bytree=param_dict['feature_fraction'],
                                            reg_alpha=param_dict['lambda_l1'],
                                            reg_lambda=param_dict['lambda_l2'],
                                            random_state=self._seed,
                                            n_jobs=-1,
                                            silent=True,
                                            importance_type='split',
                                            max_bin=param_dict['max_bin'])

        lgb_classifier.fit(X=data_x,
                           y=data_y,
                           sample_weight=None,
                           init_score=None,
                           eval_set=None,
                           eval_names=None,
                           eval_sample_weight=None,
                           eval_class_weight=None,
                           eval_init_score=None,
                           eval_metric=None,
                           # early_stopping_rounds=self._early_stopping_rounds,
                           verbose=True,
                           feature_name='auto',
                           categorical_feature='auto',
                           callbacks=None)

        return lgb_classifier


class LightGBMRegressor(SklearnPredictArrayMixin, LightGBMBase):

    def _train(self,
               data_x: pd.DataFrame,
               data_y: Union[np.ndarray, None],
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is not None

        param_dict = hyper_params.params_dict
        lgb_regressor = lgb.LGBMRegressor(boosting_type=self._boosting_type,
                                            num_leaves=param_dict['num_leaves'],
                                            max_depth=param_dict['max_depth'],
                                            learning_rate=param_dict['learning_rate'],
                                            n_estimators=param_dict['n_estimators'],
                                            subsample_for_bin=200000,
                                            objective=LightGBMObjective.REGRESSION.value,
                                            # is_unbalanced=param_dict['is_unbalanced'],
                                            # scale_pos_weight=param_dict['scale_pos_weight'],
                                            min_split_gain=param_dict['min_gain_to_split'],
                                            min_child_weight=param_dict['min_sum_hessian_in_leaf'],
                                            min_child_samples=param_dict['min_data_in_leaf'],
                                            subsample=param_dict['bagging_fraction'],
                                            subsample_freq=param_dict['bagging_freq'],
                                            colsample_bytree=param_dict['feature_fraction'],
                                            reg_alpha=param_dict['lambda_l1'],
                                            reg_lambda=param_dict['lambda_l2'],
                                            random_state=self._seed,
                                            n_jobs=-1,
                                            silent=True,
                                            importance_type='split',
                                            max_bin=param_dict['max_bin'])

        lgb_regressor.fit(X=data_x,
                           y=data_y,
                           sample_weight=None,
                           init_score=None,
                           eval_set=None,
                           eval_names=None,
                           eval_sample_weight=None,
                           eval_init_score=None,
                           eval_metric=None,
                           # early_stopping_rounds=self._early_stopping_rounds,
                           verbose=True,
                           feature_name='auto',
                           categorical_feature='auto',
                           callbacks=None)

        return lgb_regressor
