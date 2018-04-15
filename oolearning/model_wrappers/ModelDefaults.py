from typing import Union


from oolearning.enums.CategoricalEncoding import CategoricalEncoding
from oolearning.enums.DummyClassifierStrategy import DummyClassifierStrategy
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_wrappers.AdaBoost import AdaBoostRegressor, AdaBoostRegressorHP, AdaBoostClassifier, \
    AdaBoostClassifierHP
from oolearning.model_wrappers.CartDecisionTree import CartDecisionTreeRegressor, CartDecisionTreeHP, \
    CartDecisionTreeClassifier
from oolearning.model_wrappers.DummyClassifier import DummyClassifier
from oolearning.model_wrappers.ElasticNetRegressor import ElasticNetRegressor, ElasticNetRegressorHP
from oolearning.model_wrappers.GradientBoosting import GradientBoostingRegressor, \
    GradientBoostingRegressorHP, GradientBoostingClassifier, GradientBoostingClassifierHP
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid
from oolearning.model_wrappers.LassoRegressor import LassoRegressor, LassoRegressorHP
from oolearning.model_wrappers.LinearRegressor import LinearRegressor
from oolearning.model_wrappers.LogisticClassifier import LogisticClassifier, LogisticClassifierHP
from oolearning.model_wrappers.RandomForest import RandomForestRegressor, RandomForestHP, \
    RandomForestClassifier
from oolearning.model_wrappers.RidgeRegressor import RidgeRegressor, RidgeRegressorHP
from oolearning.model_wrappers.SoftmaxLogisticClassifier import SoftmaxLogisticClassifier, SoftmaxLogisticHP
from oolearning.model_wrappers.SupportVectorMachines import SvmLinearClassifier, SvmLinearClassifierHP, \
    SvmPolynomialClassifier, SvmPolynomialClassifierHP, SvmLinearRegressor, SvmLinearRegressorHP, \
    SvmPolynomialRegressor, SvmPolynomialRegressorHP
from oolearning.transformers.CenterScaleTransformer import CenterScaleTransformer
from oolearning.transformers.DummyEncodeTransformer import DummyEncodeTransformer
from oolearning.transformers.ImputationTransformer import ImputationTransformer
from oolearning.transformers.PolynomialFeaturesTransformer import PolynomialFeaturesTransformer
from oolearning.transformers.RemoveCorrelationsTransformer import RemoveCorrelationsTransformer
from oolearning.transformers.RemoveNZVTransformer import RemoveNZVTransformer


# noinspection PyTypeChecker,PyPep8Naming
class ModelDefaults:

    ###################################################
    # Regression Models
    ###################################################
    @staticmethod
    def get_LinearRegressor(degrees: Union[int, None]=None) -> ModelInfo:
        model_wrapper = LinearRegressor()
        description = type(model_wrapper).__name__
        # TODO: fill out rest of recommended transformations, verify order
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           RemoveNZVTransformer(),
                           RemoveCorrelationsTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

        if degrees is not None:
            # assumes center/scaling data should be done before the polynomial transformation
            description = '{0}_{1}_{2}'.format(description, 'polynomial', str(degrees))
            transformations.append(PolynomialFeaturesTransformer(degrees=degrees))

        return ModelInfo(description=description,
                         model=model_wrapper,
                         transformations=transformations,
                         hyper_params=None,
                         hyper_params_grid=None)

    @staticmethod
    def get_RidgeRegressor(degrees: Union[int, None]=None) -> ModelInfo:
        return ModelDefaults._ridge_lasso_elastic_helper(model_wrapper=RidgeRegressor(),
                                                         hyper_params=RidgeRegressorHP(),
                                                         degrees=degrees,
                                                         params_dict={'alpha': [0, 0.01, 0.1, 1]})

    @staticmethod
    def get_LassoRegressor(degrees: Union[int, None] = None) -> ModelInfo:
        return ModelDefaults._ridge_lasso_elastic_helper(model_wrapper=LassoRegressor(),
                                                         hyper_params=LassoRegressorHP(),
                                                         degrees=degrees,
                                                         params_dict={'alpha': [0, 0.01, 0.1, 1]})

    @staticmethod
    def get_ElasticNetRegressor(degrees: Union[int, None] = None) -> ModelInfo:
        return ModelDefaults._ridge_lasso_elastic_helper(model_wrapper=ElasticNetRegressor(),
                                                         hyper_params=ElasticNetRegressorHP(),
                                                         degrees=degrees,
                                                         params_dict={'alpha': [0.01, 0.1, 1],
                                                                      'l1_ratio': [0, 0.5, 1]})

    @staticmethod
    def get_CartDecisionTreeRegressor() -> ModelInfo:
        model_wrapper = CartDecisionTreeRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=CartDecisionTreeHP(criterion='mse'),
                         hyper_params_grid=dict(max_depth=[3, 10, 30]))

    @staticmethod
    def get_RandomForestRegressor(number_of_features: int) -> ModelInfo:
        model_wrapper = RandomForestRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         #  https://stackoverflow.com/questions/24715230/can-sklearn-random-forest-directly-handle-categorical-features?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                         transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=RandomForestHP(criterion='gini'),
                         hyper_params_grid=dict(max_features=[int(round(number_of_features**(1/2.0))),
                                                              int(round(number_of_features/2)),
                                                              number_of_features],
                                                n_estimators=[10, 100, 500],
                                                min_samples_leaf=[1, 50, 100]))

    @staticmethod
    def get_SvmLinearRegressor() -> ModelInfo:
        model_wrapper = SvmLinearRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          CenterScaleTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=SvmLinearRegressorHP(),
                         hyper_params_grid={'epsilon': [0, 0.1, 1, 3],
                                            'penalty_c': [0.001, 0.01, 0.1, 1000]})

    @staticmethod
    def get_SvmPolynomialRegressor() -> ModelInfo:
        model_wrapper = SvmPolynomialRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          CenterScaleTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=SvmPolynomialRegressorHP(),
                         hyper_params_grid={'degree': [2, 3],
                                            'epsilon': [0, 0.1, 1, 3],
                                            'penalty_c': [0.001, 0.01, 0.1, 1000]})

    @staticmethod
    def get_AdaBoostRegressor() -> ModelInfo:
        model_wrapper = AdaBoostRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=None,
                         hyper_params=AdaBoostRegressorHP(),
                         hyper_params_grid=dict(max_depth=[3, 10, 30],
                                                n_estimators=[10, 100, 500],
                                                learning_rate=[0.1, 0.5, 1]))

    @staticmethod
    def get_GradientBoostingRegressor() -> ModelInfo:
        model_wrapper = GradientBoostingRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=GradientBoostingRegressorHP(),
                         hyper_params_grid=dict(learning_rate=[0.1, 0.5, 1],
                                                n_estimators=[50, 100, 5000],
                                                max_depth=[1, 5, 9],
                                                min_samples_leaf=[1, 10, 20]))

    @staticmethod
    def get_regression_models(number_of_features):
        """
        returns a list of ModelInfos containing all available regression models.
        :return:
        """
        return [ModelDefaults.get_LinearRegressor(),
                ModelDefaults.get_LinearRegressor(degrees=2),
                ModelDefaults.get_LinearRegressor(degrees=3),
                ModelDefaults.get_RidgeRegressor(),
                ModelDefaults.get_RidgeRegressor(degrees=2),
                ModelDefaults.get_RidgeRegressor(degrees=3),
                ModelDefaults.get_LassoRegressor(),
                ModelDefaults.get_LassoRegressor(degrees=2),
                ModelDefaults.get_LassoRegressor(degrees=3),
                ModelDefaults.get_ElasticNetRegressor(),
                ModelDefaults.get_ElasticNetRegressor(degrees=2),
                ModelDefaults.get_ElasticNetRegressor(degrees=3),
                ModelDefaults.get_CartDecisionTreeRegressor(),
                ModelDefaults.get_RandomForestRegressor(number_of_features=number_of_features),
                ModelDefaults.get_SvmLinearRegressor(),
                ModelDefaults.get_SvmPolynomialRegressor(),
                ModelDefaults.get_AdaBoostRegressor(),
                ModelDefaults.get_GradientBoostingRegressor()]

    @staticmethod
    def _ridge_lasso_elastic_helper(model_wrapper, hyper_params, degrees, params_dict):
        description = type(model_wrapper).__name__
        # TODO: fill out rest of recommended transformations, verify order
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           RemoveNZVTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

        if degrees is not None:
            # assumes center/scaling data should be done before the polynomial transformation
            description = '{0}_{1}_{2}'.format(description, 'polynomial', str(degrees))
            transformations.append(PolynomialFeaturesTransformer(degrees=degrees))

        return ModelInfo(description=description,
                         model=model_wrapper,
                         transformations=transformations,
                         hyper_params=hyper_params,
                         hyper_params_grid=HyperParamsGrid(params_dict=params_dict))

    ###################################################
    # Classification Models
    ###################################################
    @staticmethod
    def get_DummyClassifier(strategy: DummyClassifierStrategy) -> ModelInfo:
        model_wrapper = DummyClassifier(strategy=strategy)
        return ModelInfo(description='{0}_{1}'.format(type(model_wrapper).__name__, strategy.value),
                         model=model_wrapper,
                         transformations=None,
                         hyper_params=None,
                         hyper_params_grid=None)

    @staticmethod
    def get_LogisticClassifier(degrees: Union[int, None]=None) -> ModelInfo:

        model_wrapper = LogisticClassifier()
        description = type(model_wrapper).__name__
        # TODO: fill out rest of recommended transformations, verify order
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           RemoveNZVTransformer(),
                           RemoveCorrelationsTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

        if degrees is not None:
            # assumes center/scaling data should be done before the polynomial transformation
            description = '{0}_{1}_{2}'.format(description, 'polynomial', str(degrees))
            transformations.append(PolynomialFeaturesTransformer(degrees=degrees))

        return ModelInfo(description=description,
                         model=model_wrapper,
                         transformations=transformations,
                         hyper_params=LogisticClassifierHP(),
                         hyper_params_grid={'penalty': ['l1', 'l2'],
                                            'C': [0.001, 0.01, 0.1, 1, 100, 1000]})

    @staticmethod
    def get_CartDecisionTreeClassifier() -> ModelInfo:
        model_wrapper = CartDecisionTreeClassifier()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=None,
                         hyper_params=RandomForestHP(),
                         hyper_params_grid=dict(criterion='gini',
                                                max_depth=[3, 10, 30]))

    @staticmethod
    def get_RandomForestClassifier(number_of_features: int) -> ModelInfo:
        model_wrapper = RandomForestClassifier()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         # https://stackoverflow.com/questions/24715230/can-sklearn-random-forest-directly-handle-categorical-features?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
                         transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=RandomForestHP(),
                         hyper_params_grid=dict(criterion='gini',
                                                max_features=[int(round(number_of_features ** (1 / 2.0))),
                                                              int(round(number_of_features / 2)),
                                                              number_of_features],
                                                n_estimators=[10, 100, 500],
                                                min_samples_leaf=[1, 50, 100]))

    @staticmethod
    def get_SvmLinearClassifier() -> ModelInfo:
        model_wrapper = SvmLinearClassifier()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          CenterScaleTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=SvmLinearClassifierHP(),
                         hyper_params_grid={  # The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads
                                            # to coef_ vectors that are sparse.
                                            'penalty': ['l2'],
                                            #  a smaller C value leads to a wider street but more margin
                                            #  violations (HOML pg 148)
                                            'penalty_c': [0.001, 0.01, 0.1, 1, 100, 1000]})

    @staticmethod
    def get_SvmPolynomialClassifier() -> ModelInfo:
        model_wrapper = SvmPolynomialClassifier()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          CenterScaleTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=SvmPolynomialClassifierHP(),
                         hyper_params_grid={'degree': [2, 3],
                                            'coef0': [0, 1, 10],
                                            #  a smaller C value leads to a wider street but more margin
                                            #  violations (HOML pg 148)
                                            'penalty_c': [0.001, 0.1, 100, 1000]})

    @staticmethod
    def get_AdaBoostClassifier() -> ModelInfo:
        model_wrapper = AdaBoostClassifier()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=None,
                         hyper_params=AdaBoostClassifierHP(),
                         hyper_params_grid=dict(max_depth=[3, 10, 30],
                                                n_estimators=[10, 100, 500],
                                                learning_rate=[0.1, 0.5, 1]))

    @staticmethod
    def get_GradientBoostingClassifier() -> ModelInfo:
        model_wrapper = GradientBoostingClassifier()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=None,
                         hyper_params=GradientBoostingClassifierHP(),
                         hyper_params_grid=dict(learning_rate=[0.1, 0.5, 1],
                                                n_estimators=[50, 100, 5000],
                                                max_depth=[1, 5, 9],
                                                min_samples_leaf=[1, 10, 20]))

    # noinspection SpellCheckingInspection
    @staticmethod
    def get_twoclass_classification_models(number_of_features):
        """
        returns a list of ModelInfos containing all available classification models.
        :return:
        """
        return [ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.MOST_FREQUENT),
                ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.PRIOR),
                ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.STRATIFIED),
                ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.UNIFORM),
                ModelDefaults.get_LogisticClassifier(),
                ModelDefaults.get_LogisticClassifier(degrees=2),
                ModelDefaults.get_LogisticClassifier(degrees=3),
                ModelDefaults.get_SvmLinearClassifier(),
                ModelDefaults.get_SvmPolynomialClassifier(),
                ModelDefaults.get_CartDecisionTreeClassifier(),
                ModelDefaults.get_RandomForestClassifier(number_of_features=number_of_features),
                ModelDefaults.get_AdaBoostClassifier(),
                ModelDefaults.get_GradientBoostingClassifier()]

    ###################################################
    # Multi-Classification Models
    ###################################################
    @staticmethod
    def get_SoftmaxLogisticClassifier(degrees: Union[int, None] = None) -> ModelInfo:

        model_wrapper = SoftmaxLogisticClassifier()
        description = type(model_wrapper).__name__
        # TODO: fill out rest of recommended transformations, verify order
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           RemoveNZVTransformer(),
                           RemoveCorrelationsTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

        if degrees is not None:
            # assumes center/scaling data should be done before the polynomial transformation
            description = '{0}_{1}_{2}'.format(description, 'polynomial', str(degrees))
            transformations.append(PolynomialFeaturesTransformer(degrees=degrees))

        return ModelInfo(description=description,
                         model=model_wrapper,
                         transformations=transformations,
                         hyper_params=SoftmaxLogisticHP(),
                         hyper_params_grid={'C': [0.001, 0.01, 0.1, 1, 100, 1000]})

    @staticmethod
    def get_multiclass_classification_models(number_of_features):
        """
        returns a list of ModelInfos containing all available classification models.
        :return:
        """
        return [ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.MOST_FREQUENT),
                ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.PRIOR),
                ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.STRATIFIED),
                ModelDefaults.get_DummyClassifier(strategy=DummyClassifierStrategy.UNIFORM),
                ModelDefaults.get_SoftmaxLogisticClassifier(),
                ModelDefaults.get_CartDecisionTreeClassifier(),
                ModelDefaults.get_RandomForestClassifier(number_of_features=number_of_features),
                ModelDefaults.get_AdaBoostClassifier(),
                ModelDefaults.get_GradientBoostingClassifier()]
