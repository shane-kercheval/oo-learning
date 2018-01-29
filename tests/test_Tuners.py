import os
import unittest
from math import isclose
from os import remove

import dill as pickle
import matplotlib.pyplot as plt

from oolearning import *

from tests.MockClassificationModelWrapper import MockClassificationModelWrapper
from tests.MockEvaluator import MockEvaluator
from tests.MockHyperParams import MockHyperParams
from tests.MockResampler import MockResampler
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic,SpellCheckingInspection,PyTypeChecker
class TunerTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_HyperParamsGrid(self):
        expected_dict = {'criterion': 'gini',
                         'max_features': [3, 6, 11],
                         'n_estimators': [10, 100, 500],
                         'min_samples_leaf': [1, 50, 100]}
        params_dict = ModelDefaults.hyper_params_random_forest_classification(number_of_features=11)
        assert params_dict == expected_dict

        grid = HyperParamsGrid(params_dict=params_dict)
        assert grid.params_grid.shape[0] == 3**3
        assert grid.params_grid.shape[1] == len(params_dict)
        assert all(grid.params_grid.criterion == 'gini')
        assert all(grid.params_grid.max_features.unique() == params_dict['max_features'])
        assert all(grid.params_grid.n_estimators.unique() == params_dict['n_estimators'])
        assert all(grid.params_grid.min_samples_leaf.unique() == params_dict['min_samples_leaf'])

    @unittest.skip("test takes several minutes")
    def test_ModelTuner_RandomForest_classification(self):
        """
        I want to keep this to run manually in the future, but running the Tuner/Resampler for a RandomForest
        model takes several minutes, and is not practical. I've saved the tune_results to a file and use a
        Mock Resampler to Mock out this test.
        """
        data = TestHelper.get_titanic_data()

        train_data = data
        train_data_y = train_data.Survived
        train_data = train_data.drop('Survived', axis=1)

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        evaluator_list = [KappaEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          SensitivityEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          SpecificityEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          ErrorRateTwoClassEvaluator(positive_category=1, negative_category=0, threshold=0.5)]

        cache_directory = TestHelper.ensure_test_directory('data/test_Tuners/cached_test_models/test_ModelTuner_RandomForest_classification')  # noqa
        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=RandomForestMW(),
                                                                      model_transformations=transformations,
                                                                      evaluators=evaluator_list),
                           hyper_param_object=RandomForestHP(),
                           persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
        params_dict = ModelDefaults.hyper_params_random_forest_classification(number_of_features=len(columns))
        grid = HyperParamsGrid(params_dict=params_dict)

        import time
        t0 = time.time()
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)
        t1 = time.time()
        total_execution_time = t1 - t0
        print(total_execution_time)

        assert len(tuner.results._tune_results_objects) == 27
        assert all([isinstance(x, ResamplerResults)
                    for x in tuner.results._tune_results_objects.resampler_object])

        # evaluator columns should be in the same order as specificied in the list
        assert all(tuner.results.tune_results.columns.values == ['criterion', 'max_features', 'n_estimators',
                                                                 'min_samples_leaf', 'kappa_mean',
                                                                 'kappa_st_dev', 'sensitivity_mean',
                                                                 'sensitivity_st_dev', 'specificity_mean',
                                                                 'specificity_st_dev', 'ErrorRate_mean',
                                                                 'ErrorRate_st_dev'])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelTuner_classification_mock.pkl'))  # noqa
        with open(file, 'wb') as output:
            pickle.dump(tuner.results, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            tune_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.tune_results,
                                                      data_frame2=tuner.results.tune_results)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.time_results,
                                                      data_frame2=tuner.results.time_results)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

        assert all(tuner.results.sorted_best_models.index.values ==
                   [24, 21, 15, 9, 12, 18, 6, 3, 0, 23, 20, 17, 16, 14, 13, 11, 4, 26, 10, 7, 5, 1, 19, 8, 2,
                    22, 25])

    def test_ModelTuner_mock_classification(self):
        """
        This unit test uses a Mock Resampler (and other necessary mocks), because testing an actual tuner
        would take too long. The Mock Resampler simply looks up the kappa/sensitivity/specificity values from
        a previously run (actual) Resampler, based on each iteration's hyper-parameters.
        So in theory, each line of the ModelTuner should still be tested, it is just relaying on fake data
        fro the Mock Resampler.
        """
        data = TestHelper.get_titanic_data()

        train_data = data
        train_data_y = train_data.Survived
        train_data = train_data.drop('Survived', axis=1)

        evaluators = [MockEvaluator(metric_name='kappa', better_than=lambda x, y: x > y),
                      MockEvaluator(metric_name='sensitivity', better_than=lambda x, y: x > y),
                      MockEvaluator(metric_name='specificity', better_than=lambda x, y: x > y),
                      MockEvaluator(metric_name='ErrorRate', better_than=lambda x, y: x < y)]

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        tuner = ModelTuner(resampler=MockResampler(model=MockClassificationModelWrapper(data_y=data.Survived),
                                                   model_transformations=transformations,
                                                   evaluators=evaluators),
                           hyper_param_object=MockHyperParams())

        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
        assert len(columns) == 24
        params_dict = ModelDefaults.hyper_params_random_forest_classification(number_of_features=len(columns))
        grid = HyperParamsGrid(params_dict=params_dict)
        assert len(grid.params_grid == 27)
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)

        assert len(tuner.results._tune_results_objects) == 27
        assert all([isinstance(x, ResamplerResults)
                    for x in tuner.results._tune_results_objects.resampler_object])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelTuner_classification_mock.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(tuner.results, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            tune_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.tune_results,
                                                      data_frame2=tuner.results.tune_results)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

        assert all(tuner.results.time_results.columns.values ==
                   ['criterion', 'max_features', 'n_estimators', 'min_samples_leaf', 'execution_time'])
        assert len(tuner.results.time_results) == len(tuner.results.tune_results)
        assert tuner.results.time_results.isnull().sum().sum() == 0

        ######################################################################################################
        # Test Best Model
        ######################################################################################################
        # test the correct order of best models (i.e. highest to lowest kappa)
        assert all(tuner.results.sorted_best_models.index.values ==
                   [24, 21, 15, 9, 12, 18, 6, 3, 0, 23, 20, 17, 16, 14, 13, 11, 4, 26, 10, 7, 5, 1, 19, 8, 2,
                    22, 25])
        assert isclose(tuner.results.best_model.kappa_mean, 0.59010255445858673)
        assert tuner.results.best_hyper_params == {'criterion': 'gini',
                                                   'max_features': 24,
                                                   'n_estimators': 500,
                                                   'min_samples_leaf': 1}

        ######################################################################################################
        # Test Heatmap
        ######################################################################################################
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Tuners/test_ModelTuner_mock_classification_get_heatmap.png'))  # noqa
        assert os.path.isfile(file)
        remove(file)
        assert os.path.isfile(file) is False
        tuner.results.get_heatmap()
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

    def test_tuner_with_no_hyper_params(self):
        data = TestHelper.get_cement_data()

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop('strength', axis=1)

        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=RegressionMW(),
                                                                      model_transformations=ModelDefaults.transformations_regression(),  # noqa
                                                                      evaluators=[RmseEvaluator(),
                                                                                  MaeEvaluator()],
                                                                      folds=5,
                                                                      repeats=5),
                           hyper_param_object=None)

        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=None)

        assert len(tuner.results._tune_results_objects) == 1

        assert isinstance(tuner.results._tune_results_objects.resampler_object[0], ResamplerResults)

        assert len(tuner.results._tune_results_objects.iloc[0].resampler_object._evaluators) == 25
        assert all([len(x) == 2 and
                    isinstance(x[0], RmseEvaluator) and
                    isinstance(x[1], MaeEvaluator)
                    for x in tuner.results._tune_results_objects.iloc[0].resampler_object._evaluators])
        assert tuner.results._tune_results_objects.iloc[0].resampler_object.num_resamples == 25
        assert tuner.results._tune_results_objects.iloc[0].resampler_object.metrics == ['RMSE', 'MAE']
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.metric_means['RMSE'], 10.459344010622544)  # noqa
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.metric_means['MAE'], 8.2855537849498742)  # noqa
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.metric_standard_deviations['RMSE'], 0.5716680069548794)  # noqa
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.metric_standard_deviations['MAE'], 0.46714447004190812)  # noqa

        assert isclose(tuner.results.tune_results.iloc[0].RMSE_mean, 10.459344010622544)
        assert isclose(tuner.results.tune_results.iloc[0].MAE_mean, 8.2855537849498742)
        assert isclose(tuner.results.tune_results.iloc[0].RMSE_st_dev, 0.5716680069548794)
        assert isclose(tuner.results.tune_results.iloc[0].MAE_st_dev, 0.46714447004190812)

        assert tuner.results.time_results.isnull().sum().sum() == 0

        ######################################################################################################
        # Test Best Model
        # note, there is only 1 model, so we are more/less just testing out that calling these methods works
        ######################################################################################################
        assert all(tuner.results.sorted_best_models.index.values == [0])
        assert isclose(tuner.results.best_model.RMSE_mean, 10.459344010622544)
        assert tuner.results._hyper_params is None
        assert tuner.results.best_hyper_params is None
        ######################################################################################################
        # Test Heatmap
        ######################################################################################################
        assert tuner.results.get_heatmap() is None
