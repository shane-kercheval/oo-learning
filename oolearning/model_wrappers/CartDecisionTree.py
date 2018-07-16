from typing import Union

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictProbabilityMixin, \
    SklearnPredictArrayMixin
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase


class CartDecisionTreeHP(HyperParamsBase):
    def __init__(self,
                 criterion: str='gini',
                 splitter: str='best',
                 max_depth: Union[int, None]=None,
                 min_samples_split: Union[int, float]=2,
                 min_samples_leaf: Union[int, float]=1,
                 min_weight_fraction_leaf: float=0.0,
                 max_leaf_nodes: Union[int, None]=None,
                 max_features: Union[int, float, str, None]=None):
        """
        See http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        :param criterion: Supported `criterion` values
            - for classifiers are 'gini' & 'entropy';
            - for regressors are 'mse' (Mean Squared Error) and 'mae' (Mean Absolute Error)
            - default value is 'gini' (used for classification)
        """
        super().__init__()

        self._is_regression = None

        criterion = criterion.lower()

        if criterion == 'gini' or criterion == 'entropy':
            self._is_regression = False
        elif criterion == 'mse' or criterion == 'mae':
            self._is_regression = True
        else:
            raise ValueError('invalid criterion')

        self._params_dict = dict(criterion=criterion,
                                 splitter=splitter,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_leaf_nodes=max_leaf_nodes,
                                 max_features=max_features)

    @property
    def is_regression(self):
        return self._is_regression


class CartDecisionTreeClassifier(SklearnPredictProbabilityMixin, ModelWrapperBase):
    def __init__(self, seed: int=42):
        super().__init__()
        self._seed = seed

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: CartDecisionTreeHP) -> object:
        assert hyper_params is not None
        assert isinstance(hyper_params, CartDecisionTreeHP)
        assert not hyper_params.is_regression

        param_dict = hyper_params.params_dict

        tree = DecisionTreeClassifier(criterion=param_dict['criterion'],
                                      splitter=param_dict['splitter'],
                                      max_depth=param_dict['max_depth'],
                                      min_samples_split=param_dict['min_samples_split'],
                                      min_samples_leaf=param_dict['min_samples_leaf'],
                                      min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'],
                                      max_leaf_nodes=param_dict['max_leaf_nodes'],
                                      max_features=param_dict['max_features'],
                                      random_state=self._seed)
        tree.fit(data_x, data_y)
        return tree


class CartDecisionTreeRegressor(SklearnPredictArrayMixin, ModelWrapperBase):
    def __init__(self, seed: int=42):
        super().__init__()
        self._seed = seed

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: CartDecisionTreeHP) -> object:
        assert hyper_params is not None
        assert isinstance(hyper_params, CartDecisionTreeHP)
        assert hyper_params.is_regression

        param_dict = hyper_params.params_dict

        tree = DecisionTreeRegressor(criterion=param_dict['criterion'],
                                     splitter=param_dict['splitter'],
                                     max_depth=param_dict['max_depth'],
                                     min_samples_split=param_dict['min_samples_split'],
                                     min_samples_leaf=param_dict['min_samples_leaf'],
                                     min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'],
                                     max_leaf_nodes=param_dict['max_leaf_nodes'],
                                     max_features=param_dict['max_features'],
                                     random_state=self._seed)
        tree.fit(data_x, data_y)
        return tree
