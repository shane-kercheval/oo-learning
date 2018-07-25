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
        RandomForestClassifier model takes several minutes, and is not practical. I've saved the
        resampled_stats to a file and use a Mock Resampler to Mock out this test.
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
                           model_persistence_manager=LocalCacheManager(cache_directory=cache_directory),
                           parallelization_cores=0)

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

        assert os.path.isdir(cache_directory)
        assert len(tuner.results._tune_results_objects) == 27
        assert all([isinstance(x, ResamplerResults)
                    for x in tuner.results._tune_results_objects.resampler_object])

        # evaluator columns should be in the same order as specificied in the list
        assert all(tuner.results.resampled_stats.columns.values == ['criterion', 'max_features',
                                                                    'n_estimators', 'min_samples_leaf',
                                                                    'kappa_mean', 'kappa_st_dev', 'kappa_cv',
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
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampled_stats,
                                                      data_frame2=tuner.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampler_times,
                                                      data_frame2=tuner.results.resampler_times)
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
                           hyper_param_object=MockHyperParams(),
                           parallelization_cores=0)

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
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampled_stats,
                                                      data_frame2=tuner.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

        assert all(tuner.results.resampler_times.columns.values == ['criterion', 'max_features', 'n_estimators', 'min_samples_leaf', 'execution_time'])  # noqa
        assert len(tuner.results.resampler_times) == len(tuner.results.resampled_stats)
        assert tuner.results.resampler_times.isnull().sum().sum() == 0

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

        for index in range(len(tuner.results.resampled_stats)):
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_means['kappa'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['kappa_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_standard_deviations['kappa'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['kappa_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_coefficients_of_variation['kappa'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['kappa_cv'])

            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_means['sensitivity'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['sensitivity_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_standard_deviations['sensitivity'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['sensitivity_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_coefficients_of_variation['sensitivity'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['sensitivity_cv'])

            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_means['specificity'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['specificity_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_standard_deviations['specificity'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['specificity_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_coefficients_of_variation['specificity'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['specificity_cv'])

            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_means['error_rate'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['error_rate_mean'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_standard_deviations['error_rate'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['error_rate_st_dev'])
            assert isclose(tuner.results._tune_results_objects.iloc[index].resampler_object.score_coefficients_of_variation['error_rate'],  # noqa
                           tuner.results.resampled_stats.iloc[index]['error_rate_cv'])

        ######################################################################################################
        # Test Heatmap
        ######################################################################################################
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_heatmap.png',
                              lambda: tuner.results.plot_resampled_stats())

        ######################################################################################################
        # Test Box-Plots
        ######################################################################################################
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_kappa.png',
                              lambda: tuner.results.plot_resampled_scores(metric=Metric.KAPPA))
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_sens.png',
                              lambda: tuner.results.plot_resampled_scores(metric=Metric.SENSITIVITY))
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_spec.png',
                              lambda: tuner.results.plot_resampled_scores(metric=Metric.SPECIFICITY))
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_error.png',
                              lambda: tuner.results.plot_resampled_scores(metric=Metric.ERROR_RATE))

        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_kappa_string.png',  # noqa
                              lambda: tuner.results.plot_resampled_scores(score_name='kappa'))

        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_boxplot_kappa_xlims_one_ste.png',  # noqa
                              lambda: tuner.results.plot_resampled_scores(metric=Metric.KAPPA,
                                                                          x_axis_limits=(0.3, 0.8),
                                                                          show_one_ste_rule=True))

        x_axis = 'max_features'
        line = 'n_estimators'
        grid = 'min_samples_leaf'
        metric = Metric.KAPPA
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_3.png',  # noqa
                              lambda: tuner.results.plot_hyper_params_profile(metric=metric, x_axis=x_axis, line=line, grid=grid))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_2.png',  # noqa
                              lambda: tuner.results.plot_hyper_params_profile(metric=metric, x_axis=x_axis, line=line, grid=None))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_1.png',  # noqa
                              lambda: tuner.results.plot_hyper_params_profile(metric=metric, x_axis=x_axis, line=None, grid=None))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_1_ne.png',  # noqa
                              lambda: tuner.results.plot_hyper_params_profile(metric=metric, x_axis='n_estimators', line=None, grid=None))  # noqa
        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_1_msl.png',  # noqa
                              lambda: tuner.results.plot_hyper_params_profile(metric=metric, x_axis='min_samples_leaf', line=None, grid=None))  # noqa

        TestHelper.check_plot('data/test_Tuners/test_ModelTuner_mock_classification_get_profile_hyper_params_3_string.png',  # noqa
                              lambda: tuner.results.plot_hyper_params_profile(metric=metric, x_axis=x_axis, line=line, grid=grid))  # noqa

        self.assertRaises(AssertionError, lambda: tuner.results.plot_hyper_params_profile(metric=metric, x_axis=x_axis, line=None, grid=grid))  # noqa

    def test_ModelTuner_GradientBoosting_classification(self):

        data = TestHelper.get_titanic_data()

        train_data = data
        train_data_y = train_data.Survived
        train_data = train_data.drop(columns='Survived')

        ######################################################################################################
        # Build from scratch, cache models and Resampler results; then, second time, use the Resampler cache
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        evaluator_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
                          SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa

        model_cache_directory = TestHelper.ensure_test_directory('data/test_Tuners/cached_test_models/test_ModelTuner_GradientBoostingClassifier')  # noqa
        resampler_cache_directory = TestHelper.ensure_test_directory('data/test_Tuners/temp_cache/')
        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=GradientBoostingClassifier(),
                                                                      transformations=transformations,
                                                                      scores=evaluator_list),
                           hyper_param_object=GradientBoostingClassifierHP(),
                           model_persistence_manager=LocalCacheManager(cache_directory=model_cache_directory),
                           resampler_persistence_manager=LocalCacheManager(cache_directory=resampler_cache_directory,  # noqa
                                                                           sub_directory='tune_test'),
                           parallelization_cores=-1)

        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
        params_dict = dict(max_features=[int(round(len(columns) ** (1 / 2.0))),
                                         # int(round(len(columns) / 2)),
                                         len(columns)],
                           n_estimators=[10,
                                         # 100,
                                         500],
                           min_samples_leaf=[1,
                                             # 50,
                                             100])
        grid = HyperParamsGrid(params_dict=params_dict)

        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)

        # assert tuner.total_tune_time < 25  # Non-Parallelization: ~26 seconds; Parallelization: ~7 seconds

        assert os.path.isdir(model_cache_directory)

        # check files for model cache
        expected_file = 'repeat{0}_fold{1}_GradientBoostingClassifier_lossdeviance_learning_rate0.1_n_estimators{2}_max_depth3_min_samples_split2_min_samples_leaf{3}_max_features{4}_subsample1.0.pkl'  # noqa
        for fold_index in range(5):
            for repeat_index in range(5):
                for index, row in grid.params_grid.iterrows():
                    local_hyper_params = row.to_dict()
                    assert os.path.isfile(os.path.join(model_cache_directory,
                                                       expected_file.format(fold_index,
                                                                            repeat_index,
                                                                            local_hyper_params['n_estimators'],  # noqa
                                                                            local_hyper_params['min_samples_leaf'],  # noqa
                                                                            local_hyper_params['max_features'])))  # noqa

        # check files for resampler cash
        expected_file = 'resampler_results_GradientBoostingClassifier_lossdeviance_learning_rate0.1_n_estimators{0}_max_depth3_min_samples_split2_min_samples_leaf{1}_max_features{2}_subsample1.0.pkl'  # noqa
        for index, row in grid.params_grid.iterrows():
            local_hyper_params = row.to_dict()
            assert os.path.isfile(os.path.join(resampler_cache_directory,
                                               'tune_test',
                                               expected_file.format(local_hyper_params['n_estimators'],
                                                                    local_hyper_params['min_samples_leaf'],
                                                                    local_hyper_params['max_features'])))

        assert len(tuner.results._tune_results_objects) == len(grid.params_grid)
        assert all([isinstance(x, ResamplerResults)
                    for x in tuner.results._tune_results_objects.resampler_object])

        assert tuner.results.best_hyper_params == {'max_features': 24, 'n_estimators': 500, 'min_samples_leaf': 100}  # noqa

        # for each row in the underlying results dataframe, make sure the hyper-parameter values in the row
        # correspond to the same hyper_param values found in the Resampler object
        for index in range(len(tuner.results._tune_results_objects)):
            assert tuner.results._tune_results_objects.iloc[index].max_features == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['max_features']  # noqa
            assert tuner.results._tune_results_objects.iloc[index].n_estimators == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['n_estimators']  # noqa
            assert tuner.results._tune_results_objects.iloc[index].min_samples_leaf == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['min_samples_leaf']  # noqa

        resampler_scores = tuner.results._tune_results_objects.iloc[0].resampler_object.score_means
        expected_scores = {'kappa': 0.5239954603575802, 'sensitivity': 0.5187339582751682, 'specificity': 0.9639047970435495, 'error_rate': 0.20571868388646114}  # noqa

        assert resampler_scores.keys() == expected_scores.keys()
        assert all([isclose(x, y) for x, y in zip(resampler_scores.values(), expected_scores.values())])

        # evaluator columns should be in the same order as specificied in the list
        assert all(tuner.results.resampled_stats.columns.values == ['max_features',
                                                                    'n_estimators', 'min_samples_leaf',
                                                                    'kappa_mean', 'kappa_st_dev', 'kappa_cv',
                                                                    'sensitivity_mean', 'sensitivity_st_dev',
                                                                    'sensitivity_cv', 'specificity_mean',
                                                                    'specificity_st_dev', 'specificity_cv',
                                                                    'error_rate_mean', 'error_rate_st_dev',
                                                                    'error_rate_cv'])

        assert all(tuner.results.resampler_times.max_features.values == [5, 5,  5,  5, 24, 24, 24, 24])
        assert all(tuner.results.resampler_times.n_estimators.values == [10, 10, 500, 500, 10, 10, 500, 500])
        assert all(tuner.results.resampler_times.min_samples_leaf.values == [1, 100, 1, 100, 1, 100, 1, 100])
        # assert all(tuner.results.resampler_times.execution_time.values == ['7 seconds', '7 seconds', '15 seconds', '13 seconds', '7 seconds', '7 seconds', '18 seconds', '16 seconds'])  # noqa

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelTuner_GradientBoosting_results.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(tuner.results, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            tune_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampled_stats,
                                                      data_frame2=tuner.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

        assert all(tuner.results.sorted_best_models.index.values == [7, 2, 6, 3, 4, 0, 5, 1])
        shutil.rmtree(model_cache_directory)

        ######################################################################################################
        # Same, but with Resampler results cache
        ######################################################################################################
        # transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
        #                    CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
        #                    ImputationTransformer(),
        #                    DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        #
        # evaluator_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
        #                   SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
        #                   SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
        #                   ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa

        tuner_cached = ModelTuner(resampler=RepeatedCrossValidationResampler(model=GradientBoostingClassifier(),  # noqa
                                                                      transformations=None,  # diff
                                                                      scores=[]),  # diff
                           hyper_param_object=GradientBoostingClassifierHP(),
                           # including model_persistence_manager but it shouldn't be used (and cached models
                           # don't exist any longer
                           model_persistence_manager=LocalCacheManager(cache_directory=model_cache_directory),
                           resampler_persistence_manager=LocalCacheManager(cache_directory=resampler_cache_directory,  # noqa
                                                                           sub_directory='tune_test'),
                           parallelization_cores=-1)

        tuner_cached.tune(data_x=None, data_y=None, params_grid=grid)

        assert tuner_cached.total_tune_time < 1  # should be super quick with only 8 cached files to load

        assert len(tuner_cached.results._tune_results_objects) == len(grid.params_grid)
        assert all([isinstance(x, ResamplerResults)
                    for x in tuner_cached.results._tune_results_objects.resampler_object])

        assert tuner_cached.results.best_hyper_params == {'max_features': 24, 'n_estimators': 500, 'min_samples_leaf': 100}  # noqa

        # for each row in the underlying results dataframe, make sure the hyper-parameter values in the row
        # correspond to the same hyper_param values found in the Resampler object
        for index in range(len(tuner_cached.results._tune_results_objects)):
            assert tuner_cached.results._tune_results_objects.iloc[index].max_features == tuner_cached.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['max_features']  # noqa
            assert tuner_cached.results._tune_results_objects.iloc[index].n_estimators == tuner_cached.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['n_estimators']  # noqa
            assert tuner_cached.results._tune_results_objects.iloc[index].min_samples_leaf == tuner_cached.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['min_samples_leaf']  # noqa

        tuner_cached_scores = tuner_cached.results._tune_results_objects.iloc[0].resampler_object.score_means
        expected_scores = {'kappa': 0.5239954603575802, 'sensitivity': 0.5187339582751682, 'specificity': 0.9639047970435495, 'error_rate': 0.20571868388646114}  # noqa

        assert tuner_cached_scores.keys() == expected_scores.keys()
        assert all([isclose(x, y) for x, y in zip(tuner_cached_scores.values(), expected_scores.values())])

        # evaluator columns should be in the same order as specificied in the list
        assert all(tuner_cached.results.resampled_stats.columns.values == ['max_features', 'n_estimators', 'min_samples_leaf', 'kappa_mean', 'kappa_st_dev', 'kappa_cv', 'sensitivity_mean', 'sensitivity_st_dev', 'sensitivity_cv', 'specificity_mean', 'specificity_st_dev', 'specificity_cv', 'error_rate_mean', 'error_rate_st_dev', 'error_rate_cv'])  # noqa

        assert all(tuner_cached.results.resampler_times.max_features.values == [5, 5,  5,  5, 24, 24, 24, 24])
        assert all(tuner_cached.results.resampler_times.n_estimators.values == [10, 10, 500, 500, 10, 10, 500, 500])  # noqa
        assert all(tuner_cached.results.resampler_times.min_samples_leaf.values == [1, 100, 1, 100, 1, 100, 1, 100])  # noqa
        # assert all(tuner_cached.results.resampler_times.execution_time.values == ['7 seconds', '7 seconds', '15 seconds', '13 seconds', '7 seconds', '7 seconds', '18 seconds', '16 seconds'])  # noqa

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelTuner_GradientBoosting_results.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(tuner_cached.results, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            tune_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampled_stats,
                                                      data_frame2=tuner_cached.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
                                                      data_frame2=tuner_cached.results.sorted_best_models)

        assert all(tuner_cached.results.sorted_best_models.index.values == [7, 2, 6, 3, 4, 0, 5, 1])
        shutil.rmtree(resampler_cache_directory)

    def test_tuner_float_int_param_combos(self):
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

        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=XGBoostClassifier(),
                                                                      transformations=transformations,
                                                                      scores=evaluator_list,
                                                                      folds=5,
                                                                      repeats=2),
                           hyper_param_object=XGBoostTreeHP(objective=XGBObjective.BINARY_LOGISTIC),
                           parallelization_cores=-1)

        params_dict = dict(colsample_bytree=[0.7, 1.0],
                           subsample=[0.75, 1.0],
                           max_depth=[6, 9])
        grid = HyperParamsGrid(params_dict=params_dict)

        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)
        assert tuner.results.best_hyper_params == {'colsample_bytree': 0.7, 'subsample': 1.0, 'max_depth': 6}
        assert isinstance(tuner.results.best_hyper_params['colsample_bytree'], float)
        assert isinstance(tuner.results.best_hyper_params['subsample'], float)
        # bug fix where this was changed to a float because we were using .iloc
        assert isinstance(tuner.results.best_hyper_params['max_depth'], np.int64)  # hmm.. ideally `int`?

        assert len(tuner.results._tune_results_objects) == len(grid.params_grid)
        assert all([isinstance(x, ResamplerResults)
                    for x in tuner.results._tune_results_objects.resampler_object])

        # for each row in the underlying results dataframe, make sure the hyper-parameter values in the row
        # correspond to the same hyper_param values found in the Resampler object
        for index in range(len(tuner.results._tune_results_objects)):
            assert tuner.results._tune_results_objects.iloc[index].colsample_bytree == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['colsample_bytree']  # noqa
            assert tuner.results._tune_results_objects.iloc[index].subsample == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['subsample']  # noqa
            assert tuner.results._tune_results_objects.iloc[index].max_depth == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['max_depth']  # noqa

        tuner_scores = tuner.results._tune_results_objects.iloc[0].resampler_object.score_means
        expected_scores = {'kappa': 0.5939055543285305, 'sensitivity': 0.7026889336045273, 'specificity': 0.8802720680336996, 'error_rate': 0.18753105999861713}  # noqa

        assert tuner_scores.keys() == expected_scores.keys()
        assert all([isclose(x, y) for x, y in zip(tuner_scores.values(), expected_scores.values())])

        # evaluator columns should be in the same order as specificied in the list
        assert all(tuner.results.resampled_stats.columns.values == ['colsample_bytree', 'subsample',
                                                                    'max_depth', 'kappa_mean',
                                                                    'kappa_st_dev', 'kappa_cv',
                                                                    'sensitivity_mean',
                                                                    'sensitivity_st_dev', 'sensitivity_cv',
                                                                    'specificity_mean',
                                                                    'specificity_st_dev', 'specificity_cv',
                                                                    'error_rate_mean',
                                                                    'error_rate_st_dev', 'error_rate_cv'])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelTuner_XGB_results.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(tuner.results, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            tune_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampled_stats,
                                                      data_frame2=tuner.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

        assert all(tuner.results.sorted_best_models.index.values == [2, 4, 1, 3, 6, 5, 0, 7])

    def test_tuner_with_no_hyper_params(self):
        data = TestHelper.get_cement_data()

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop(columns='strength')

        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=LinearRegressorSK(),
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
        assert tuner.results._tune_results_objects.iloc[0].resampler_object.score_names == ['RMSE', 'MAE']
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.score_means['RMSE'], 10.459344010622544)  # noqa
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.score_means['MAE'], 8.2855537849498742)  # noqa
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.score_standard_deviations['RMSE'], 0.5716680069548794)  # noqa
        assert isclose(tuner.results._tune_results_objects.iloc[0].resampler_object.score_standard_deviations['MAE'], 0.46714447004190812)  # noqa

        assert isclose(tuner.results.resampled_stats.iloc[0].RMSE_mean, 10.459344010622544)
        assert isclose(tuner.results.resampled_stats.iloc[0].MAE_mean, 8.2855537849498742)
        assert isclose(tuner.results.resampled_stats.iloc[0].RMSE_st_dev, 0.5716680069548794)
        assert isclose(tuner.results.resampled_stats.iloc[0].MAE_st_dev, 0.46714447004190812)

        assert tuner.results.resampler_times.isnull().sum().sum() == 0

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
        assert tuner.results.plot_resampled_stats() is None
        assert tuner.results.plot_resampled_scores(metric=Metric.KAPPA) is None

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
        tuner = ModelTuner(resampler=resampler,
                           hyper_param_object=RandomForestHP(),
                           parallelization_cores=0)

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
