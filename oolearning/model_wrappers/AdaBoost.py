from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor as SkAdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictClassifierMixin, \
    SklearnPredictRegressorMixin


class AdaBoostClassifierHP(HyperParamsBase):
    # noinspection SpellCheckingInspection
    def __init__(self,
                 max_depth: Union[int, None]=None,
                 n_estimators: int=50,
                 learning_rate: float=1.0,
                 algorithm: str='SAMME.R'):
        """
        for more info, see
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
        """
        super().__init__()

        self._params_dict = dict(max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 algorithm=algorithm)


class AdaBoostClassifier(SklearnPredictClassifierMixin, ModelWrapperBase):
    def __init__(self, random_state: int=42):
        super().__init__()
        self._random_state = random_state

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: AdaBoostClassifierHP) -> object:
        assert hyper_params is not None
        param_dict = hyper_params.params_dict
        tree = SkAdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=param_dict['max_depth']),
                                    n_estimators=param_dict['n_estimators'],
                                    learning_rate=param_dict['learning_rate'],
                                    algorithm=param_dict['algorithm'],
                                    random_state=self._random_state)
        tree.fit(data_x, data_y)
        return tree


class AdaBoostRegressorHP(HyperParamsBase):
    def __init__(self,
                 max_depth: Union[int, None]=None,
                 n_estimators: int=50,
                 learning_rate: float=1.0,
                 loss: str='linear'):
        """
        for more info, see
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
        """
        super().__init__()

        self._params_dict = dict(max_depth=max_depth,
                                 n_estimators=n_estimators,
                                 learning_rate=learning_rate,
                                 loss=loss)


class AdaBoostRegressor(SklearnPredictRegressorMixin, ModelWrapperBase):
    def __init__(self, random_state: int=42):
        super().__init__()
        self._random_state = random_state

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: AdaBoostRegressorHP) -> object:
        assert hyper_params is not None
        param_dict = hyper_params.params_dict
        tree = SkAdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=param_dict['max_depth']),
                                   n_estimators=param_dict['n_estimators'],
                                   learning_rate=param_dict['learning_rate'],
                                   loss=param_dict['loss'],
                                   random_state=self._random_state)
        tree.fit(data_x, data_y)
        return tree
