from typing import Union

from oolearning.hyper_params.HyperParamsBase import HyperParamsBase


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

        if self._is_regression is None:
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
