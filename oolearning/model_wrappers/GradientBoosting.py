from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier as SkGradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor as SkGradientBoostingRegressor

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictProbabilityMixin, \
    SklearnPredictArrayMixin


class GradientBoostingClassifierHP(HyperParamsBase):
    def __init__(self,
                 loss: str='deviance',
                 learning_rate: float=0.1,
                 n_estimators: int=100,
                 max_depth: int=3,
                 min_samples_split: Union[int, float]=2,
                 min_samples_leaf: Union[int, float]=1,
                 max_features: Union[int, float, str, None]=None,
                 subsample: Union[float, None]=1.0):
        """
        for more info, see
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        """
        super().__init__()
        self._params_dict = dict(loss=loss,
                                 learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features,
                                 subsample=subsample)


class GradientBoostingClassifier(SklearnPredictProbabilityMixin, ModelWrapperBase):
    def __init__(self, seed: int=42):
        super().__init__()
        self._seed = seed

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: GradientBoostingClassifierHP) -> object:
        assert hyper_params is not None
        assert isinstance(hyper_params, GradientBoostingClassifierHP)
        param_dict = hyper_params.params_dict
        tree = SkGradientBoostingClassifier(loss=param_dict['loss'],
                                            learning_rate=param_dict['learning_rate'],
                                            n_estimators=param_dict['n_estimators'],
                                            max_depth=param_dict['max_depth'],
                                            min_samples_split=param_dict['min_samples_split'],
                                            min_samples_leaf=param_dict['min_samples_leaf'],
                                            max_features=param_dict['max_features'],
                                            subsample=param_dict['subsample'],
                                            random_state=self._seed)
        tree.fit(data_x, data_y)
        return tree


class GradientBoostingRegressorHP(HyperParamsBase):
    def __init__(self,
                 loss: str='ls',
                 learning_rate: float=0.1,
                 n_estimators: int=100,
                 max_depth: int=3,
                 min_samples_split: Union[int, float]=2,
                 min_samples_leaf: Union[int, float]=1,
                 max_features: Union[int, float, str, None]=None,
                 subsample: Union[float, None]=1.0):
        """
        for more info, see
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
        """
        super().__init__()
        self._params_dict = dict(loss=loss,
                                 learning_rate=learning_rate,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 max_features=max_features,
                                 subsample=subsample)


class GradientBoostingRegressor(SklearnPredictArrayMixin, ModelWrapperBase):
    def __init__(self, seed: int=42):
        super().__init__()
        self._seed = seed

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: GradientBoostingRegressorHP) -> object:
        assert hyper_params is not None
        assert isinstance(hyper_params, GradientBoostingRegressorHP)
        param_dict = hyper_params.params_dict
        tree = SkGradientBoostingRegressor(loss=param_dict['loss'],
                                           learning_rate=param_dict['learning_rate'],
                                           n_estimators=param_dict['n_estimators'],
                                           max_depth=param_dict['max_depth'],
                                           min_samples_split=param_dict['min_samples_split'],
                                           min_samples_leaf=param_dict['min_samples_leaf'],
                                           max_features=param_dict['max_features'],
                                           subsample=param_dict['subsample'],
                                           random_state=self._seed)
        tree.fit(data_x, data_y)
        return tree
