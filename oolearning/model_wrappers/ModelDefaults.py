from typing import List

from oolearning.model_wrappers.LinearRegressor import LinearRegressor
from oolearning.enums.CategoricalEncoding import CategoricalEncoding
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.transformers.CenterScaleTransformer import CenterScaleTransformer
from oolearning.transformers.DummyEncodeTransformer import DummyEncodeTransformer
from oolearning.transformers.ImputationTransformer import ImputationTransformer
from oolearning.transformers.PolynomialFeaturesTransformer import PolynomialFeaturesTransformer
from oolearning.transformers.RemoveCorrelationsTransformer import RemoveCorrelationsTransformer
from oolearning.transformers.RemoveNZPTransformer import RemoveNZPTransformer
from oolearning.transformers.TransformerBase import TransformerBase


# noinspection PyTypeChecker,PyPep8Naming
class ModelDefaults:

    ###################################################
    # Regression Models
    ###################################################
    @staticmethod
    def get_LinearRegressor() -> ModelInfo:
        return ModelInfo(description='linear_regression',
                         model_wrapper=LinearRegressor(),
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.DUMMY),
                                          CenterScaleTransformer(),
                                          RemoveNZPTransformer(),
                                          RemoveCorrelationsTransformer()],
                         hyper_params=None,
                         hyper_params_grid=None)

    @staticmethod
    def get_LinearRegressor_polynomial(degrees: int) -> ModelInfo:
        return ModelInfo(description='linear_regression_polynomial_' + str(degrees),
                         model_wrapper=LinearRegressor(),
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.DUMMY),
                                          CenterScaleTransformer(),
                                          RemoveNZPTransformer(),
                                          RemoveCorrelationsTransformer(),
                                          PolynomialFeaturesTransformer(degrees=degrees)],
                         hyper_params=None,
                         hyper_params_grid=None)

    @property
    def regression_models(self):
        """
        returns a list of ModelInfos containing all available regression models.
        :return:
        """
        return [ModelDefaults.get_LinearRegressor(),
                ModelDefaults.get_LinearRegressor_polynomial(degrees=2),
                ModelDefaults.get_LinearRegressor_polynomial(degrees=3)]

    ###################################################
    # Classification Models
    ###################################################

    ###################################################
    # TODO: CONVERT
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
