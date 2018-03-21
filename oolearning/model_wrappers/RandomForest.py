import numpy as np
import pandas as pd
from matplotlib import figure
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from typing import Union

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


# noinspection PyPropertyDefinition
class RandomForestFI(FittedInfoBase):
    @property
    def results_summary(self) -> object:
        pass

    @property
    def summary_stats(self) -> dict:
        pass

    @property
    def warnings(self):
        pass

    @property
    def feature_importance(self) -> dict:
        pass

    @property
    def graph(self) -> figure.Figure:
        pass


class RandomForestHP(HyperParamsBase):
    """
            sklearn is the underlying model used, so the parameters (with the exception of num_features)
            correspond 1-to-1 with:
                http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
                http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

            Rather than duplicate, please refer to sklearn's documentation.

            Note:
                - n_estimators is defaulted to 500 in this object, whereas sklearn's default is 10
                - n_jobs is defaulted to -1 in this object, whereas sklearn's default is 1
                - random_state is defaulted to 42 in this object, whereas sklearn's default is None

            ------------------
            Notes on hyper-parameters:

            - increasing `max_features` generally increases model performance but decreases speed (1)
            - increasing `n_estimators` generally increases model performance but decreases speed (1)

            (1) https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/
            -----------------
    """
    def __init__(self,
                 criterion: str='gini',
                 num_features: int=None,
                 max_features: Union[int, float, str]=None,
                 n_estimators: int=500,
                 max_depth: int=None,
                 min_samples_split: Union[int, float]=2,
                 min_samples_leaf: Union[int, float]=1,
                 min_weight_fraction_leaf: float=0.0,
                 max_leaf_nodes: int=None,
                 min_impurity_decrease: float=0,
                 bootstrap: bool=True,
                 oob_score: bool=False,
                 n_jobs: int=-1,
                 random_state: int=42):
        """
        :param criterion: Supported `criterion` values
            - for classifiers are 'gini' & 'entropy';
            - for regressors are 'mse' (Mean Squared Error) and 'mae' (Mean Absolute Error)
            - default value is 'gini' (used for classification)
        :param num_features: if num_features is set,
            max_features will be set to
                - the square root of num_features for classification problems, and
                - 1/3 of the number of features for regression problems (APM pg 199)
            Regression vs Classification is determined from the criterion passed in.
        """
        super().__init__()

        # either num_features or max_features, but not both
        # note, both can be None, because sklearn supports None value for max_features
        criterion = criterion.lower()

        self._is_regression = None
        if criterion == 'gini' or criterion == 'entropy':
            self._is_regression = False
        elif criterion == 'mse' or criterion == 'mae':
            self._is_regression = True
        else:
            raise ValueError('invalid criterion')

        if num_features is not None:
            assert max_features is None
            if self._is_regression:
                max_features = int(round(num_features / 3))
            else:  # classification
                max_features = int(round(num_features**(1/2.0)))

        self._params_dict = dict(n_estimators=n_estimators,
                                 criterion=criterion,
                                 max_features=max_features,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_leaf_nodes=max_leaf_nodes,
                                 min_impurity_decrease=min_impurity_decrease,
                                 bootstrap=bootstrap,
                                 oob_score=oob_score,
                                 n_jobs=n_jobs,
                                 random_state=random_state)

    @property
    def is_regression(self):
        return self._is_regression


class RandomForest(ModelWrapperBase):
    """
    Random Forest is a small tweak on Tree Bagging where, "each time a split in a tree is considered, a
        random sample of m features is chosen as split candidates from the full set of p features. The
        split is allowed to use only one of those m features... We can think of this process as
        decorrelating the trees, thereby making the average of the resulting trees less variable and hence
        more reliable." (ISLR pg 319-320)

    A typical value is the square root of the number of features (p). "If a random Forest is built
        using m = p, then this amounts simply to bagging... Using a small value of m in building a random
        forest will typically be helpful when we have a large number of correlated features." (ISLR pg
        319-320)
    """

    def _create_fitted_info_object(self, model_object, data_x: pd.DataFrame, data_y: np.ndarray,
                                   hyper_params: HyperParamsBase = None) -> FittedInfoBase:
        return RandomForestFI(model_object=model_object,
                              feature_names=data_x.columns.values.tolist(),
                              hyper_params=hyper_params)

    # noinspection PyMethodOverriding
    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray, hyper_params: RandomForestHP) -> object:
        assert hyper_params is not None

        param_dict = hyper_params.params_dict
        if hyper_params.is_regression:
            rf_model = RandomForestRegressor(n_estimators=param_dict['n_estimators'],
                                             criterion=param_dict['criterion'],
                                             max_features=param_dict['max_features'],
                                             max_depth=param_dict['max_depth'],
                                             min_samples_split=param_dict['min_samples_split'],
                                             min_samples_leaf=param_dict['min_samples_leaf'],
                                             min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'],
                                             max_leaf_nodes=param_dict['max_leaf_nodes'],
                                             min_impurity_decrease=param_dict['min_impurity_decrease'],
                                             bootstrap=param_dict['bootstrap'],
                                             oob_score=param_dict['oob_score'],
                                             n_jobs=param_dict['n_jobs'],
                                             random_state=param_dict['random_state'])
        else:  # Classification Problem
            #  n_jobs: The number of jobs to run in parallel for both fit and predict. If -1, then the number
            # of jobs is set to the number of cores.
            rf_model = RandomForestClassifier(n_estimators=param_dict['n_estimators'],
                                              criterion=param_dict['criterion'],
                                              max_features=param_dict['max_features'],
                                              max_depth=param_dict['max_depth'],
                                              min_samples_split=param_dict['min_samples_split'],
                                              min_samples_leaf=param_dict['min_samples_leaf'],
                                              min_weight_fraction_leaf=param_dict['min_weight_fraction_leaf'],
                                              max_leaf_nodes=param_dict['max_leaf_nodes'],
                                              min_impurity_decrease=param_dict['min_impurity_decrease'],
                                              bootstrap=param_dict['bootstrap'],
                                              oob_score=param_dict['oob_score'],
                                              n_jobs=param_dict['n_jobs'],
                                              random_state=param_dict['random_state'])

        # Train the model to take the training features and learn how they relate
        # to the training y (the species)
        rf_model.fit(data_x, data_y)

        return rf_model

    # noinspection PyUnresolvedReferences
    # noinspection SpellCheckingInspection
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> pd.DataFrame:
        if self.fitted_info.hyper_params.is_regression:
            return model_object.predict(data_x)
        else:
            # `predict_proba` returns the probabilities (rows) for each class (columns);
            # transform to dataframe
            predictions = pd.DataFrame(model_object.predict_proba(data_x))
            predictions.columns = model_object.classes_

            return predictions
