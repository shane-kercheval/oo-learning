from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor as SkAdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictProbabilityMixin, \
    SklearnPredictArrayMixin


class AdaBoostClassifierHP(HyperParamsBase):
    # noinspection SpellCheckingInspection
    def __init__(self,
                 # Adaboost-specific hyper-params
                 n_estimators: int = 50,
                 learning_rate: float = 1.0,
                 algorithm: str='SAMME.R',
                 # Tree-specific hyper-params
                 criterion: str='gini',
                 splitter: str='best',
                 max_depth: Union[int, None]=None,
                 min_samples_split: Union[int, float]=2,
                 min_samples_leaf: Union[int, float]=1,
                 min_weight_fraction_leaf: float=0.,
                 max_features: Union[int, float, str, None]=None,
                 max_leaf_nodes: Union[int, None]=None,
                 min_impurity_decrease: float=0.,
                 class_weight: Union[dict, List[dict], str, None]=None,
                 presort: bool=False,
                 ):
        """
        for more info, see
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
        """
        super().__init__()

        self._params_dict = dict(
            # Adaboost-specific hyper-params
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            # Tree-specific hyper-params
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            class_weight=class_weight,
            presort=presort,
        )


class AdaBoostClassifier(SklearnPredictProbabilityMixin, ModelWrapperBase):
    def __init__(self, seed: int=42):
        super().__init__()
        self._seed = seed

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: AdaBoostClassifierHP) -> object:
        assert hyper_params is not None
        assert isinstance(hyper_params, AdaBoostClassifierHP)
        param_dict = hyper_params.params_dict
        tree = SkAdaBoostClassifier(base_estimator=DecisionTreeClassifier(
                                          criterion=param_dict['criterion'],
                                          splitter=param_dict['splitter'],
                                          max_depth=param_dict['max_depth'],
                                          min_samples_split=param_dict['min_samples_split'],
                                          min_samples_leaf=param_dict['min_samples_leaf'],
                                          min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'],
                                          max_features=param_dict['max_features'],
                                          max_leaf_nodes=param_dict['max_leaf_nodes'],
                                          min_impurity_decrease=param_dict['min_impurity_decrease'],
                                          class_weight=param_dict['class_weight'],
                                          presort=param_dict['presort'],
                                          random_state=self._seed),

                                    n_estimators=param_dict['n_estimators'],
                                    learning_rate=param_dict['learning_rate'],
                                    algorithm=param_dict['algorithm'],
                                    random_state=self._seed,
                                    )
        tree.fit(data_x, data_y)
        return tree


class AdaBoostRegressorHP(HyperParamsBase):
    # noinspection SpellCheckingInspection
    def __init__(self,
                 # Adaboost-specific hyper-params
                 n_estimators: int=50,
                 learning_rate: float=1.,
                 loss: str='linear',
                 # Tree-specific hyper-params
                 criterion='mse',
                 splitter='best',
                 max_depth: Union[int, None]=None,
                 min_samples_split: Union[int, float]=2,
                 min_samples_leaf: Union[int, float]=1,
                 min_weight_fraction_leaf: float=0.,
                 max_features: Union[int, float, str, None]=None,
                 max_leaf_nodes: Union[int, None]=None,
                 min_impurity_decrease: float=0.,
                 presort: bool=False,
                 ):
        """
        for more info, see
            http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
        """
        super().__init__()

        self._params_dict = dict(
            # Adaboost-specific hyper-params
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            # Tree-specific hyper-params
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            presort=presort,
        )


class AdaBoostRegressor(SklearnPredictArrayMixin, ModelWrapperBase):
    def __init__(self, seed: int=42):
        super().__init__()
        self._seed = seed

    @property
    def feature_importance(self):
        raise NotImplementedError()

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: AdaBoostRegressorHP) -> object:
        assert hyper_params is not None
        assert isinstance(hyper_params, AdaBoostRegressorHP)
        param_dict = hyper_params.params_dict
        tree = SkAdaBoostRegressor(base_estimator=DecisionTreeRegressor(
                                        criterion=param_dict['criterion'],
                                        splitter=param_dict['splitter'],
                                        max_depth=param_dict['max_depth'],
                                        min_samples_split=param_dict['min_samples_split'],
                                        min_samples_leaf=param_dict['min_samples_leaf'],
                                        min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'],
                                        max_features=param_dict['max_features'],
                                        max_leaf_nodes=param_dict['max_leaf_nodes'],
                                        min_impurity_decrease=param_dict['min_impurity_decrease'],
                                        presort=param_dict['presort'],
                                        random_state=self._seed),
                                   n_estimators=param_dict['n_estimators'],
                                   learning_rate=param_dict['learning_rate'],
                                   loss=param_dict['loss'],
                                   random_state=self._seed)
        tree.fit(data_x, data_y)
        return tree
