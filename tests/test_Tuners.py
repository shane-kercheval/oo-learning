import os
import shutil
import unittest
from math import isclose

import dill as pickle
import numpy as np

from oolearning import *
from tests.MockClassificationModelWrapper import MockClassificationModelWrapper
from tests.MockEvaluator import MockUtilityEvaluator, MockCostEvaluator
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
        params_dict = dict(criterion='gini',
                           max_features=[int(round(11 ** (1 / 2.0))),
                                         int(round(11 / 2)),
                                         11],
                           n_estimators=[10, 100, 500],
                           min_samples_leaf=[1, 50, 100])
        assert params_dict == expected_dict
        grid = HyperParamsGrid(params_dict=params_dict)
        assert grid.params_grid.columns.values.tolist() == ['criterion', 'max_features', 'n_estimators', 'min_samples_leaf']  # noqa
        assert grid.params_grid.shape[0] == 3**3
        assert grid.params_grid.shape[1] == len(params_dict)
        assert all(grid.params_grid.criterion == 'gini')
        assert all(grid.params_grid.max_features.unique() == params_dict['max_features'])
        assert all(grid.params_grid.n_estimators.unique() == params_dict['n_estimators'])
        assert all(grid.params_grid.min_samples_leaf.unique() == params_dict['min_samples_leaf'])

        assert grid.hyper_params == ['criterion', 'max_features', 'n_estimators', 'min_samples_leaf']
        assert grid.tuned_hyper_params == ['max_features', 'n_estimators', 'min_samples_leaf']

    @unittest.skip("test takes several minutes")
    def test_ModelTuner_RandomForest_classification(self):
        """
        I want to keep this to run manually in the future, but running the Tuner/Resampler for a
        RandomForestClassifier model takes several minutes, and is not practical. I've saved the tune_results
        to a file and use a Mock Resampler to Mock out this test.
        """
        data = TestHelper.get_titanic_data()

        train_data = data
        train_data_y = train_data.Survived
        train_data = train_data.drop(columns='Survived')

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        evaluator_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
                          SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa

        cache_directory = TestHelper.ensure_test_directory('data/test_Tuners/cached_test_models/test_ModelTuner_RandomForest_classification')  # noqa
        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=RandomForestClassifier(),
                                                                      transformations=transformations,
                                                                      scores=evaluator_list),
                           hyper_param_object=RandomForestHP(),
                           persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
        params_dict = dict(criterion='gini',
                           max_features=[int(round(len(columns) ** (1 / 2.0))),
                                         int(round(len(columns) / 2)),
                                         len(columns)],
                           n_estimators=[10, 100, 500],
                           min_samples_leaf=[1, 50, 100])
        grid = HyperParamsGrid(params_dict=params_dict)

        # import time
        # t0 = time.time()
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)
        # t1 = time.time()
        # total_execution_time = t1 - t0
        # print(total_execution_time)

        assert len(tuner.results._tune_results_objects) == 27
        assert all([isinstance(x, ResamplerResults)
                    for x in tuner.results._tune_results_objects.resampler_object])

        # evaluator columns should be in the same order as specificied in the list
        assert all(tuner.results.tune_results.columns.values == ['criterion', 'max_features', 'n_estimators',
                                                                 'min_samples_leaf', 'kappa_mean',
                                                                 'kappa_st_dev', 'kappa_cv',
                                                                 'sensitivity_mean', 'sensitivity_st_dev',
                                                                 'sensitivity_cv', 'specificity_mean',
                                                                 'specificity_st_dev', 'specificity_cv',
                                                                 'error_rate_mean', 'error_rate_st_dev',
                                                                 'error_rate_cv'])

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

        assert all(tuner.results.sorted_best_models.index.values == [24, 21, 9, 15, 12, 18, 6, 3, 0, 23, 20, 17, 16, 14, 13, 11, 4, 26, 10, 7, 5, 1, 19, 8, 2, 22, 25])  # noqa
        shutil.rmtree(cache_directory)

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
        train_data = train_data.drop(columns='Survived')

        evaluators = [MockUtilityEvaluator(metric_name='kappa'),
                      MockUtilityEvaluator(metric_name='sensitivity'),
                      MockUtilityEvaluator(metric_name='specificity'),
                      MockCostEvaluator(metric_name='error_rate')]

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        tuner = ModelTuner(resampler=MockResampler(model=MockClassificationModelWrapper(data_y=data.Survived),
                                                   transformations=transformations,
                                                   scores=evaluators),
                           hyper_param_object=MockHyperParams())

        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
        assert len(columns) == 24
        params_dict = dict(criterion='gini',
                           max_features=[int(round(len(columns) ** (1 / 2.0))),
                                         int(round(len(columns) / 2)),
                                         len(columns)],
                           n_estimators=[10, 100, 500],
                           min_samples_leaf=[1, 50, 100])
        grid = HyperParamsGrid(params_dict=params_dict)

        assert len(grid.params_grid == 27)
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)

        assert len(tuner.results._tune_results_objects) == 27
        assert tuner.results.num_param_combos == 27
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
        assert all(tuner.results.sorted_best_models.index.values == [24, 21, 9, 15, 12, 18, 6, 3, 0, 23, 20, 17, 16, 14, 13, 11, 4, 26, 10, 7, 5, 1, 19, 8, 2, 22, 25])  # noqa
        assert isclose(tuner.results.best_model.kappa_mean, 0.587757500452066)
        assert tuner.results.best_hyper_params == {'criterion': 'gini',
                                                   'max_features': 24,
                                                   'n_estimators': 500,
                                                   'min_samples_leaf': 1}

        for index in range(len(tuner.results.tune_results)):
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_means['kappa'],  # noqa
                           tuner.results.tune_results.iloc[index]['kappa_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_standard_deviations['kappa'],  # noqa
                           tuner.results.tune_results.iloc[index]['kappa_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_coefficient_of_variation['kappa'],  # noqa
                           tuner.results.tune_results.iloc[index]['kappa_cv'])

            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_means['sensitivity'],  # noqa
                           tuner.results.tune_results.iloc[index]['sensitivity_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_standard_deviations['sensitivity'],  # noqa
                           tuner.results.tune_results.iloc[index]['sensitivity_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_coefficient_of_variation['sensitivity'],  # noqa
                           tuner.results.tune_results.iloc[index]['sensitivity_cv'])

            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_means['specificity'],  # noqa
                           tuner.results.tune_results.iloc[index]['specificity_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_standard_deviations['specificity'],  # noqa
                           tuner.results.tune_results.iloc[index]['specificity_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_coefficient_of_variation['specificity'],  # noqa
                           tuner.results.tune_results.iloc[index]['specificity_cv'])

            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_means['error_rate'],  # noqa
                           tuner.results.tune_results.iloc[index]['error_rate_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_standard_deviations['error_rate'],  # noqa
                           tuner.results.tune_results.iloc[index]['error_rate_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.metric_coefficient_of_variation['error_rate'],  # noqa
                           tuner.results.tune_results.iloc[index]['error_rate_cv'])

        ######################################################################################################
        # Test Heatmap
        ######################################################################################################
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_heatmap.png',  # noqa
                              lambda: tuner.results.get_heatmap())

        ######################################################################################################
        # Test Box-Plots
        ######################################################################################################
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_kappa.png',
                              lambda: tuner.results.get_cross_validation_boxplots(metric=Metric.KAPPA))
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_sens.png',
                              lambda: tuner.results.get_cross_validation_boxplots(metric=Metric.SENSITIVITY))
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_spec.png',
                              lambda: tuner.results.get_cross_validation_boxplots(metric=Metric.SPECIFICITY))
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_error.png',
                              lambda: tuner.results.get_cross_validation_boxplots(metric=Metric.ERROR_RATE))

        x_axis = 'max_features'
        line = 'n_estimators'
        grid = 'min_samples_leaf'
        metric = Metric.KAPPA
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_3.png',  # noqa
                              lambda: tuner.results.get_profile_hyper_params(metric=metric, x_axis=x_axis, line=line, grid=grid))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_2.png',  # noqa
                              lambda: tuner.results.get_profile_hyper_params(metric=metric, x_axis=x_axis, line=line, grid=None))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_1.png',  # noqa
                              lambda: tuner.results.get_profile_hyper_params(metric=metric, x_axis=x_axis, line=None, grid=None))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_1_ne.png',  # noqa
                              lambda: tuner.results.get_profile_hyper_params(metric=metric, x_axis='n_estimators', line=None, grid=None))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_1_msl.png',  # noqa
                              lambda: tuner.results.get_profile_hyper_params(metric=metric, x_axis='min_samples_leaf', line=None, grid=None))  # noqa
        self.assertRaises(AssertionError, lambda: tuner.results.get_profile_hyper_params(metric=metric, x_axis=x_axis, line=None, grid=grid))  # noqa

    def test_tuner_with_no_hyper_params(self):
        data = TestHelper.get_cement_data()

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop(columns='strength')

        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=LinearRegressor(),
                                                                      transformations=[ImputationTransformer(),  # noqa
                                                                                       DummyEncodeTransformer(CategoricalEncoding.DUMMY)],  # noqa
                                                                      scores=[RmseScore(),
                                                                              MaeScore()],
                                                                      folds=5,
                                                                      repeats=5),
                           hyper_param_object=None)

        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=None)

        assert len(tuner.results._tune_results_objects) == 1

        assert isinstance(tuner.results._tune_results_objects.resampler_object[0], ResamplerResults)

        assert len(tuner.results._tune_results_objects.iloc[0].resampler_object._scores) == 25
        assert all([len(x) == 2 and
                    isinstance(x[0], RmseScore) and
                    isinstance(x[1], MaeScore)
                    for x in tuner.results._tune_results_objects.iloc[0].resampler_object._scores])
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
        assert tuner.results._params_grid is None
        assert tuner.results.best_hyper_params is None
        ######################################################################################################
        # Test Heatmap
        ######################################################################################################
        assert tuner.results.get_heatmap() is None
        assert tuner.results.get_cross_validation_boxplots(metric=Metric.KAPPA) is None

    def test_tuner_resampler_decorators(self):
        decorator = TwoClassThresholdDecorator()
        # resampler gets the positive class from either the score directly, or the score._converter; test
        # using both score types (e.g. AucX & Kappa)
        data = TestHelper.get_titanic_data()
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.25)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data_y = data.iloc[training_indexes].Survived
        train_data = data.iloc[training_indexes].drop(columns='Survived')

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [AucRocScore(positive_class=1)]
        resampler = RepeatedCrossValidationResampler(model=RandomForestClassifier(),
                                                     transformations=transformations,
                                                     scores=score_list,
                                                     folds=2,
                                                     repeats=1,
                                                     fold_decorators=[decorator])
        tuner = ModelTuner(resampler=resampler, hyper_param_object=RandomForestHP())

        params_dict = {'criterion': 'gini', 'max_features': [5]}
        grid = HyperParamsGrid(params_dict=params_dict)
        assert len(grid.params_grid) == 1  # just need to test 1 hyper-param combination
        # just need to test the first row
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)

        # should have cloned the decorator each time, so it should not have been used
        assert len(decorator._roc_ideal_thresholds) == 0
        assert len(decorator._precision_recall_ideal_thresholds) == 0

        # resampler_decorators is a list (per hyper-param combo), of lists (multiple decorators)
        assert len(tuner.results.resampler_decorators) == 1
        assert len(tuner.results.resampler_decorators[0]) == 1
        assert isinstance(tuner.results.resampler_decorators[0][0], TwoClassThresholdDecorator)

        assert len(tuner.results.resampler_decorators_first) == 1
        assert isinstance(tuner.results.resampler_decorators_first[0], TwoClassThresholdDecorator)

        assert tuner.results.resampler_decorators[0][0] is tuner.results.resampler_decorators_first[0]

        decorator = tuner.results.resampler_decorators_first[0]

        expected_roc_thresholds = [0.37, 0.37]
        expected_precision_recall_thresholds = [0.37, 0.49]

        assert decorator.roc_ideal_thresholds == expected_roc_thresholds
        assert decorator.precision_recall_ideal_thresholds == expected_precision_recall_thresholds

        assert isclose(decorator.roc_ideal_thresholds_mean, np.mean(expected_roc_thresholds))
        assert isclose(decorator.resampled_precision_recall_mean, np.mean(expected_precision_recall_thresholds))  # noqa
        assert isclose(decorator.roc_ideal_thresholds_st_dev, np.std(expected_roc_thresholds))
        assert isclose(decorator.resampled_precision_recall_st_dev, np.std(expected_precision_recall_thresholds))  # noqa
        assert isclose(decorator.roc_ideal_thresholds_cv, round(np.std(expected_roc_thresholds) / np.mean(expected_roc_thresholds), 2))  # noqa
        assert isclose(decorator.resampled_precision_recall_cv, round(np.std(expected_precision_recall_thresholds) / np.mean(expected_precision_recall_thresholds), 2))  # noqa
