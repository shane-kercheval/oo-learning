import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from oolearning.fitted_info.RandomForestFI import RandomForestFI
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.hyper_params.RandomForestHP import RandomForestHP
from oolearning.fitted_info.FittedInfoBase import FittedInfoBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class RandomForestMW(ModelWrapperBase):
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
