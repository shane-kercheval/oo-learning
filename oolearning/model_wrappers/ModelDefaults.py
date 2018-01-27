from typing import List

from oolearning.enums.CategoricalEncoding import CategoricalEncoding
from oolearning.transformers.DummyEncodeTransformer import DummyEncodeTransformer
from oolearning.transformers.ImputationTransformer import ImputationTransformer
from oolearning.transformers.TransformerBase import TransformerBase


# noinspection PyTypeChecker
class ModelDefaults:

    ###################################################
    # Linear Regression
    ###################################################
    @staticmethod
    def hyper_params_regression() -> dict:
        return None  # no hyper-parameters for linear regression

    @staticmethod
    def transformations_regression() -> List[TransformerBase]:
        return [ImputationTransformer(),
                DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

    ###################################################
    # Logistic Regression
    ###################################################
    @staticmethod
    def hyper_params_logistic() -> dict:
        return None  # no hyper-parameters for logistic regression

    @staticmethod
    def transformations_logistic() -> List[TransformerBase]:
        return [ImputationTransformer(),
                DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

    ###################################################
    # Random Forest
    ###################################################
    @staticmethod
    def hyper_params_random_forest_classification(number_of_features) -> dict:
        return dict(criterion='gini',
                    max_features=[int(round(number_of_features**(1/2.0))),
                                  int(round(number_of_features/2)),
                                  number_of_features],
                    n_estimators=[10, 100, 500],
                    min_samples_leaf=[1, 50, 100])

    @staticmethod
    def hyper_params_random_forest_regression(number_of_features) -> dict:
        # [x for x in range(tune_length-1, 0, -1)]
        return dict(mtry=[1, 2],
                    ntree=500)

    @staticmethod
    def transformations_random_forest() -> List[TransformerBase]:
        return None
