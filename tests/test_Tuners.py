import itertools
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


class ModelDecorator(DecoratorBase):
    def __init__(self):
        self._model_list = list()

    def decorate(self, **kwargs):
        self._model_list.append(kwargs['model'])


class TransformerDecorator(DecoratorBase):
    def __init__(self):
        self._pipeline_list = list()

    def decorate(self, **kwargs):
        self._pipeline_list.append(kwargs['transformer_pipeline'])


# noinspection PyMethodMayBeStatic,SpellCheckingInspection,PyTypeChecker,PyUnresolvedReferences
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

        assert grid.param_names == ['criterion', 'max_features', 'n_estimators', 'min_samples_leaf']
        assert grid.tuned_hyper_params == ['max_features', 'n_estimators', 'min_samples_leaf']

    @unittest.skip("test takes several minutes")
    def test_GridSearchModelTuner_RandomForest_classification(self):
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

        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
        params_dict = dict(criterion='gini',
                           max_features=[int(round(len(columns) ** (1 / 2.0))),
                                         int(round(len(columns) / 2)),
                                         len(columns)],
                           n_estimators=[10, 100, 500],
                           min_samples_leaf=[1, 50, 100])
        grid = HyperParamsGrid(params_dict=params_dict)

        cache_directory = TestHelper.ensure_test_directory('data/test_Tuners/cached_test_models/test_ModelTuner_RandomForest_classification')  # noqa
        tuner = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=RandomForestClassifier(),  # noqa
                                                                                transformations=transformations,  # noqa
                                                                                scores=evaluator_list),
                                     hyper_param_object=RandomForestHP(),
                                     params_grid=grid,
                                     model_persistence_manager=LocalCacheManager(cache_directory=cache_directory),  # noqa
                                     parallelization_cores=0)

        # import time
        # t0 = time.time()
        tuner.tune(data_x=train_data, data_y=train_data_y)
        TestHelper.save_string(tuner.results,
                               'data/test_Tuners/test_GridSearchModelTuner_RandomForest_classification_string.txt')  # noqa

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

    def test_GridSearchModelTuner_mock_classification(self):
        """
        This unit test uses a Mock Resampler (and other necessary mocks), because testing an actual tuner
        would take too long. The Mock Resampler simply looks up the kappa/sensitivity/specificity values from
        a previously run (actual) Resampler, based on each iteration's hyper-parameters.
        So in theory, each line of the GridSearchModelTuner should still be tested, it is just relaying on
        fake data from the Mock Resampler.
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

        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
        assert len(columns) == 24
        params_dict = dict(criterion='gini',
                           max_features=[int(round(len(columns) ** (1 / 2.0))),
                                         int(round(len(columns) / 2)),
                                         len(columns)],
                           n_estimators=[10, 100, 500],
                           min_samples_leaf=[1, 50, 100])
        grid = HyperParamsGrid(params_dict=params_dict)

        tuner = GridSearchModelTuner(resampler=MockResampler(model=MockClassificationModelWrapper(data_y=data.Survived),  # noqa
                                                             transformations=transformations,
                                                             scores=evaluators),
                                     hyper_param_object=MockHyperParams(),
                                     params_grid=grid,
                                     parallelization_cores=0)

        assert len(grid.params_grid == 27)
        tuner.tune(data_x=train_data, data_y=train_data_y)
        TestHelper.save_string(tuner.results,
                               'data/test_Tuners/test_GridSearchModelTuner_mock_classification_string.txt')  # noqa

        assert len(tuner.results._tune_results_objects) == 27
        assert tuner.results.number_of_cycles == 27
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

    # def test_GridSearchModelTuner_GradientBoosting_classification(self):
    #
    #     data = TestHelper.get_titanic_data()
    #
    #     train_data = data
    #     train_data_y = train_data.Survived
    #     train_data = train_data.drop(columns='Survived')
    #
    #     ######################################################################################################
    #     # Build from scratch, cache models and Resampler results; then, second time, use the Resampler cache
    #     ######################################################################################################
    #     transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
    #                        CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
    #                        ImputationTransformer(),
    #                        DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
    #
    #     evaluator_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
    #                       SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
    #                       SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
    #                       ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa
    #
    #     columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=train_data)
    #     params_dict = dict(max_features=[int(round(len(columns) ** (1 / 2.0))),
    #                                      # int(round(len(columns) / 2)),
    #                                      len(columns)],
    #                        n_estimators=[10,
    #                                      # 100,
    #                                      500],
    #                        min_samples_leaf=[1,
    #                                          # 50,
    #                                          100])
    #     grid = HyperParamsGrid(params_dict=params_dict)
    #
    #     model_cache_directory = TestHelper.ensure_test_directory('data/test_Tuners/cached_test_models/test_ModelTuner_GradientBoostingClassifier')  # noqa
    #     resampler_cache_directory = TestHelper.ensure_test_directory('data/test_Tuners/temp_cache/')
    #
    #     resampler = RepeatedCrossValidationResampler(model=GradientBoostingClassifier(),
    #                                                  transformations=transformations,
    #                                                  scores=evaluator_list)
    #     tuner = GridSearchModelTuner(resampler=resampler,
    #                                  hyper_param_object=GradientBoostingClassifierHP(),
    #                                  params_grid=grid,
    #                                  model_persistence_manager=LocalCacheManager(cache_directory=model_cache_directory),  # noqa
    #                                  resampler_persistence_manager=LocalCacheManager(cache_directory=resampler_cache_directory,  # noqa
    #                                                                        sub_directory='tune_test'),
    #                                  parallelization_cores=-1)
    #
    #     tuner.tune(data_x=train_data, data_y=train_data_y)
    #     TestHelper.save_string(tuner.results,
    #                            'data/test_Tuners/test_GridSearchModelTuner_GradientBoosting_classification_string.txt')  # noqa
    #
    #     # assert tuner.total_tune_time < 25  # Non-Parallelization: ~26 seconds; Parallelization: ~7 seconds
    #
    #     assert os.path.isdir(model_cache_directory)
    #
    #     # check files for model cache
    #     expected_file = 'repeat{0}_fold{1}_GradientBoostingClassifier_lossdeviance_learning_rate0.1_n_estimators{2}_max_depth3_min_samples_split2_min_samples_leaf{3}_max_features{4}_subsample1.0.pkl'  # noqa
    #     for fold_index in range(5):
    #         for repeat_index in range(5):
    #             for index, row in grid.params_grid.iterrows():
    #                 local_hyper_params = row.to_dict()
    #                 assert os.path.isfile(os.path.join(model_cache_directory,
    #                                                    expected_file.format(fold_index,
    #                                                                         repeat_index,
    #                                                                         local_hyper_params['n_estimators'],  # noqa
    #                                                                         local_hyper_params['min_samples_leaf'],  # noqa
    #                                                                         local_hyper_params['max_features'])))  # noqa
    #
    #     # check files for resampler cash
    #     expected_file = 'resampler_results_GradientBoostingClassifier_lossdeviance_learning_rate0.1_n_estimators{0}_max_depth3_min_samples_split2_min_samples_leaf{1}_max_features{2}_subsample1.0.pkl'  # noqa
    #     for index, row in grid.params_grid.iterrows():
    #         local_hyper_params = row.to_dict()
    #         assert os.path.isfile(os.path.join(resampler_cache_directory,
    #                                            'tune_test',
    #                                            expected_file.format(local_hyper_params['n_estimators'],
    #                                                                 local_hyper_params['min_samples_leaf'],
    #                                                                 local_hyper_params['max_features'])))
    #
    #     assert len(tuner.results._tune_results_objects) == len(grid.params_grid)
    #     assert all([isinstance(x, ResamplerResults)
    #                 for x in tuner.results._tune_results_objects.resampler_object])
    #
    #     assert tuner.results.best_hyper_params == {'max_features': 24, 'n_estimators': 500, 'min_samples_leaf': 100}  # noqa
    #
    #     # for each row in the underlying results dataframe, make sure the hyper-parameter values in the row
    #     # correspond to the same hyper_param values found in the Resampler object
    #     for index in range(len(tuner.results._tune_results_objects)):
    #         assert tuner.results._tune_results_objects.iloc[index].max_features == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['max_features']  # noqa
    #         assert tuner.results._tune_results_objects.iloc[index].n_estimators == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['n_estimators']  # noqa
    #         assert tuner.results._tune_results_objects.iloc[index].min_samples_leaf == tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['min_samples_leaf']  # noqa
    #
    #     resampler_scores = tuner.results._tune_results_objects.iloc[0].resampler_object.score_means
    #     expected_scores = {'kappa': 0.5239954603575802, 'sensitivity': 0.5187339582751682, 'specificity': 0.9639047970435495, 'error_rate': 0.20571868388646114}  # noqa
    #
    #     assert resampler_scores.keys() == expected_scores.keys()
    #     assert all([isclose(x, y) for x, y in zip(resampler_scores.values(), expected_scores.values())])
    #
    #     # evaluator columns should be in the same order as specificied in the list
    #     assert all(tuner.results.resampled_stats.columns.values == ['max_features',
    #                                                                 'n_estimators', 'min_samples_leaf',
    #                                                                 'kappa_mean', 'kappa_st_dev', 'kappa_cv',
    #                                                                 'sensitivity_mean', 'sensitivity_st_dev',
    #                                                                 'sensitivity_cv', 'specificity_mean',
    #                                                                 'specificity_st_dev', 'specificity_cv',
    #                                                                 'error_rate_mean', 'error_rate_st_dev',
    #                                                                 'error_rate_cv'])
    #
    #     assert all(tuner.results.resampler_times.max_features.values == [5, 5,  5,  5, 24, 24, 24, 24])
    #     assert all(tuner.results.resampler_times.n_estimators.values == [10, 10, 500, 500, 10, 10, 500, 500])
    #     assert all(tuner.results.resampler_times.min_samples_leaf.values == [1, 100, 1, 100, 1, 100, 1, 100])
    #     # assert all(tuner.results.resampler_times.execution_time.values == ['7 seconds', '7 seconds', '15 seconds', '13 seconds', '7 seconds', '7 seconds', '18 seconds', '16 seconds'])  # noqa
    #
    #     TestHelper.save_df(tuner.results.resampled_stats,
    #                        'data/test_Tuners/test_ModelTuner_GradientBoosting_results__resampled_stats.csv')  # noqa
    #
    #     TestHelper.save_df(tuner.results.sorted_best_models,
    #                        'data/test_Tuners/test_ModelTuner_GradientBoosting_results__sorted_best_models.csv')  # noqa
    #
    #
    #     file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelTuner_GradientBoosting_results.pkl'))  # noqa
    #     # with open(file, 'wb') as output:
    #     #     pickle.dump(tuner.results, output, pickle.HIGHEST_PROTOCOL)
    #     with open(file, 'rb') as saved_object:
    #         tune_results = pickle.load(saved_object)
    #
    #         assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampled_stats,
    #                                                   data_frame2=tuner.results.resampled_stats)
    #         assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
    #                                                   data_frame2=tuner.results.sorted_best_models)
    #
    #     assert all(tuner.results.sorted_best_models.index.values == [7, 2, 6, 3, 4, 0, 5, 1])
    #     shutil.rmtree(model_cache_directory)
    #
    #     ######################################################################################################
    #     # Same, but with Resampler results cache
    #     ######################################################################################################
    #     # transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
    #     #                    CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
    #     #                    ImputationTransformer(),
    #     #                    DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
    #     #
    #     # evaluator_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
    #     #                   SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
    #     #                   SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
    #     #                   ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa
    #
    #     tuner_cached = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=GradientBoostingClassifier(),  # noqa
    #                                                                                    transformations=None,
    #                                                                                    scores=[]),
    #                                         hyper_param_object=GradientBoostingClassifierHP(),
    #                                         params_grid=grid,
    #                                         # including model_persistence_manager but it shouldn't be used
    #                                         # (and cached models don't exist any longer
    #                                         model_persistence_manager=LocalCacheManager(cache_directory=model_cache_directory),  # noqa
    #                                         resampler_persistence_manager=LocalCacheManager(cache_directory=resampler_cache_directory,  # noqa
    #                                                                        sub_directory='tune_test'),
    #                                         parallelization_cores=-1)
    #
    #     tuner_cached.tune(data_x=None, data_y=None)
    #
    #     if not TestHelper.is_debugging():
    #         assert tuner_cached.total_tune_time < 1  # should be super quick with only 8 cached files to load
    #
    #     assert len(tuner_cached.results._tune_results_objects) == len(grid.params_grid)
    #     assert all([isinstance(x, ResamplerResults)
    #                 for x in tuner_cached.results._tune_results_objects.resampler_object])
    #
    #     assert tuner_cached.results.best_hyper_params == {'max_features': 24, 'n_estimators': 500, 'min_samples_leaf': 100}  # noqa
    #
    #     # for each row in the underlying results dataframe, make sure the hyper-parameter values in the row
    #     # correspond to the same hyper_param values found in the Resampler object
    #     for index in range(len(tuner_cached.results._tune_results_objects)):
    #         assert tuner_cached.results._tune_results_objects.iloc[index].max_features == tuner_cached.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['max_features']  # noqa
    #         assert tuner_cached.results._tune_results_objects.iloc[index].n_estimators == tuner_cached.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['n_estimators']  # noqa
    #         assert tuner_cached.results._tune_results_objects.iloc[index].min_samples_leaf == tuner_cached.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['min_samples_leaf']  # noqa
    #
    #     tuner_cached_scores = tuner_cached.results._tune_results_objects.iloc[0].resampler_object.score_means
    #     expected_scores = {'kappa': 0.5239954603575802, 'sensitivity': 0.5187339582751682, 'specificity': 0.9639047970435495, 'error_rate': 0.20571868388646114}  # noqa
    #
    #     assert tuner_cached_scores.keys() == expected_scores.keys()
    #     assert all([isclose(x, y) for x, y in zip(tuner_cached_scores.values(), expected_scores.values())])
    #
    #     # evaluator columns should be in the same order as specificied in the list
    #     assert all(tuner_cached.results.resampled_stats.columns.values == ['max_features', 'n_estimators', 'min_samples_leaf', 'kappa_mean', 'kappa_st_dev', 'kappa_cv', 'sensitivity_mean', 'sensitivity_st_dev', 'sensitivity_cv', 'specificity_mean', 'specificity_st_dev', 'specificity_cv', 'error_rate_mean', 'error_rate_st_dev', 'error_rate_cv'])  # noqa
    #
    #     assert all(tuner_cached.results.resampler_times.max_features.values == [5, 5,  5,  5, 24, 24, 24, 24])
    #     assert all(tuner_cached.results.resampler_times.n_estimators.values == [10, 10, 500, 500, 10, 10, 500, 500])  # noqa
    #     assert all(tuner_cached.results.resampler_times.min_samples_leaf.values == [1, 100, 1, 100, 1, 100, 1, 100])  # noqa
    #     # assert all(tuner_cached.results.resampler_times.execution_time.values == ['7 seconds', '7 seconds', '15 seconds', '13 seconds', '7 seconds', '7 seconds', '18 seconds', '16 seconds'])  # noqa
    #
    #     file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelTuner_GradientBoosting_results.pkl'))  # noqa
    #     # with open(file, 'wb') as output:
    #     #     pickle.dump(tuner_cached.results, output, pickle.HIGHEST_PROTOCOL)
    #     with open(file, 'rb') as saved_object:
    #         tune_results = pickle.load(saved_object)
    #         assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.resampled_stats,
    #                                                   data_frame2=tuner_cached.results.resampled_stats)
    #         assert TestHelper.ensure_all_values_equal(data_frame1=tune_results.sorted_best_models,
    #                                                   data_frame2=tuner_cached.results.sorted_best_models)
    #
    #     assert all(tuner_cached.results.sorted_best_models.index.values == [7, 2, 6, 3, 4, 0, 5, 1])
    #     shutil.rmtree(resampler_cache_directory)

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

        params_dict = dict(colsample_bytree=[0.7, 1.0],
                           subsample=[0.75, 1.0],
                           max_depth=[6, 9])
        grid = HyperParamsGrid(params_dict=params_dict)

        tuner = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=XGBoostClassifier(),
                                                                                transformations=transformations,  # noqa
                                                                                scores=evaluator_list,
                                                                                folds=5,
                                                                                repeats=2),
                                     hyper_param_object=XGBoostTreeHP(objective=XGBObjective.BINARY_LOGISTIC),
                                     params_grid=grid,
                                     # XGBoost does not work with multi-processors (even though it used to)
                                     # https: // github.com / dmlc / xgboost / issues / 4246
                                     parallelization_cores=1)  # can't parallelize with xgboost

        tuner.tune(data_x=train_data, data_y=train_data_y)
        TestHelper.save_string(tuner.results,
                               'data/test_Tuners/test_tuner_float_int_param_combos_string.txt')  # noqa

        assert tuner.results.best_hyper_params == {'colsample_bytree': 0.7, 'subsample': 0.75, 'max_depth': 6}
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

        assert all(tuner.results.sorted_best_models.index.values == [0, 1, 3, 2, 4, 6, 5, 7])

    def test_tuner_with_no_hyper_params(self):
        data = TestHelper.get_cement_data()

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop(columns='strength')

        tuner = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=LinearRegressorSK(),
                                                                                transformations=[ImputationTransformer(),  # noqa
                                                                                       DummyEncodeTransformer(CategoricalEncoding.DUMMY)],  # noqa
                                                                                scores=[RmseScore(),
                                                                              MaeScore()],
                                                                                folds=5,
                                                                                repeats=5),
                                     hyper_param_object=None,
                                     params_grid=None)

        tuner.tune(data_x=train_data, data_y=train_data_y)
        TestHelper.save_string(tuner.results,
                               'data/test_Tuners/test_tuner_with_no_hyper_params_string.txt')  # noqa

        assert len(tuner.results._tune_results_objects) == 1
        assert tuner.results.number_of_cycles == 1

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
        two_class_decorator = TwoClassThresholdDecorator()
        transformer_decorator = TransformerDecorator()
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

        params_dict = {'criterion': 'gini', 'max_features': [5]}
        grid = HyperParamsGrid(params_dict=params_dict)
        assert len(grid.params_grid) == 1  # just need to test 1 hyper-param combination

        resampler = RepeatedCrossValidationResampler(model=RandomForestClassifier(),
                                                     transformations=transformations,
                                                     scores=score_list,
                                                     folds=2,
                                                     repeats=1,
                                                     fold_decorators=[two_class_decorator,
                                                                      transformer_decorator])
        tuner = GridSearchModelTuner(resampler=resampler,
                                     hyper_param_object=RandomForestHP(),
                                     params_grid=grid,
                                     parallelization_cores=0)
        # just need to test the first row
        tuner.tune(data_x=train_data, data_y=train_data_y)
        TestHelper.save_string(tuner.results,
                               'data/test_Tuners/test_tuner_resampler_decorators_string.txt')  # noqa

        # should have cloned the decorator each time, so it should not have been used
        assert len(two_class_decorator._roc_ideal_thresholds) == 0
        assert len(two_class_decorator._precision_recall_ideal_thresholds) == 0

        # resampler_decorators is a list (per hyper-param combo), of lists (multiple decorators)
        assert len(tuner.results.resampler_decorators) == 1
        assert len(tuner.results.resampler_decorators[0]) == 2
        assert isinstance(tuner.results.resampler_decorators[0][0], TwoClassThresholdDecorator)

        assert len(tuner.results.resampler_decorators_first) == 1
        assert isinstance(tuner.results.resampler_decorators_first[0], TwoClassThresholdDecorator)

        assert tuner.results.resampler_decorators[0][0] is tuner.results.resampler_decorators_first[0]

        two_class_decorator = tuner.results.resampler_decorators_first[0]

        expected_roc_thresholds = [0.37, 0.37]
        expected_precision_recall_thresholds = [0.37, 0.49]

        assert two_class_decorator.roc_ideal_thresholds == expected_roc_thresholds
        assert two_class_decorator.precision_recall_ideal_thresholds == expected_precision_recall_thresholds

        assert isclose(two_class_decorator.roc_threshold_mean, np.mean(expected_roc_thresholds))
        assert isclose(two_class_decorator.precision_recall_threshold_mean, np.mean(expected_precision_recall_thresholds))  # noqa
        assert isclose(two_class_decorator.roc_threshold_st_dev, np.std(expected_roc_thresholds))
        assert isclose(two_class_decorator.precision_recall_threshold_st_dev, np.std(expected_precision_recall_thresholds))  # noqa
        assert isclose(two_class_decorator.roc_threshold_cv, round(np.std(expected_roc_thresholds) / np.mean(expected_roc_thresholds), 2))  # noqa
        assert isclose(two_class_decorator.precision_recall_threshold_cv, round(np.std(expected_precision_recall_thresholds) / np.mean(expected_precision_recall_thresholds), 2))  # noqa

        assert len(tuner.resampler_decorators) == tuner.results.number_of_cycles
        assert tuner.results.number_of_cycles == 1

        for index in range(tuner.results.number_of_cycles):
            local_transformer_decorator = tuner.resampler_decorators[index][1]
            pipeline_list = local_transformer_decorator._pipeline_list
            assert tuner.number_of_resamples == {len(pipeline_list)}
            for pipeline in pipeline_list:
                transformations = pipeline.transformations
                assert len(transformations) == 5  # 4 for the transformations and 1 StatelessTransformer
                assert isinstance(transformations[0], RemoveColumnsTransformer)
                assert isinstance(transformations[1], CategoricConverterTransformer)
                assert isinstance(transformations[2], ImputationTransformer)
                assert isinstance(transformations[3], DummyEncodeTransformer)
                assert isinstance(transformations[4], StatelessTransformer)
                assert all([transformation.has_executed for transformation in transformations])

    def test_GridSearchModelTuner_regression(self):
        # Test sorting & getting best model parameters works for models that are minimizing the score
        data = TestHelper.get_cement_data()

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop(columns='strength')

        ######################################################################################################
        # Build from scratch, cache models and Resampler results; then, second time, use the Resampler cache
        ######################################################################################################
        evaluator_list = [RmseScore()]  # noqa
        params_dict = dict(max_depth=[-1, 6],
                           num_leaves=[10, 50])
        grid = HyperParamsGrid(params_dict=params_dict)

        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        resampler = RepeatedCrossValidationResampler(model=LightGBMRegressor(),
                                                     transformations=None,
                                                     fold_decorators=[model_decorator,
                                                                      transformer_decorator],
                                                     scores=evaluator_list)
        tuner = GridSearchModelTuner(resampler=resampler,
                                     hyper_param_object=LightGBMHP(),
                                     params_grid=grid,
                                     # seems like tuner with LightGBM is SLOW when parallelizing
                                     parallelization_cores=0)

        tuner.tune(data_x=train_data, data_y=train_data_y)
        TestHelper.save_string(tuner.results,
                               'data/test_Tuners/test_GridSearchModelTuner_regression_string.txt')

        assert tuner.total_tune_time < 10
        assert tuner.results.best_index == 1
        assert tuner.results.best_hyper_params == {'max_depth': -1, 'num_leaves': 50}
        assert all(tuner.results.sorted_best_indexes == [1, 3, 0, 2])

        # use the ModelDecorators to check the hyper_params
        assert len(tuner.resampler_decorators) == tuner.results.number_of_cycles
        assert tuner.number_of_resamples == {5*5}

        # lets check that the "trained_params" (i.e. from the underlying model object's get_params())
        # matches what we passed into the resampler
        for decorator_index in range(len(tuner.resampler_decorators)):
            # decorator_index = 0
            decorator_list = tuner.resampler_decorators[decorator_index]
            # only passed in 1 decorator (so it is at index [0])
            assert len(decorator_list) == 2
            local_model_decorator = decorator_list[0]
            # noinspection PyUnresolvedReferences
            model_list = local_model_decorator._model_list
            assert len(model_list) == 5*5
            trained_params = [x.model_object.get_params() for x in model_list]
            for index in range(len(trained_params)):
                assert trained_params[index] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = tuner.results._tune_results_objects.resampler_object.values[decorator_index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params,
                                                   mapping={'min_gain_to_split': 'min_split_gain',
                                                            'min_sum_hessian_in_leaf': 'min_child_weight',
                                                            'min_data_in_leaf': 'min_child_samples',
                                                            'bagging_fraction': 'subsample',
                                                            'bagging_freq': 'subsample_freq',
                                                            'feature_fraction': 'colsample_bytree',
                                                            'lambda_l1': 'reg_alpha',
                                                            'lambda_l2': 'reg_lambda'},
                                                   remove_keys=['scale_pos_weight'])

            pipeline_list = decorator_list[1]._pipeline_list
            assert len(pipeline_list) == 5 * 5
            # no transformations should all be empty lists
            assert all([pipeline.transformations == [] for pipeline in pipeline_list])

    @staticmethod
    def get_LightGBMRegressor_GridSearchTuner_results():
        data = TestHelper.get_cement_data()

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop(columns='strength')

        ######################################################################################################
        # Build from scratch, cache models and Resampler results; then, second time, use the Resampler cache
        ######################################################################################################
        evaluator_list = [RmseScore(), MaeScore()]  # noqa
        params_dict = dict(max_depth=[-1, 6],
                           num_leaves=[10, 50])
        grid = HyperParamsGrid(params_dict=params_dict)

        decorator = ModelDecorator()
        tuner = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=LightGBMRegressor(),
                                                                                transformations=None,
                                                                                fold_decorators=[decorator],
                                                                                scores=evaluator_list),
                                     hyper_param_object=LightGBMHP(),
                                     params_grid=grid,
                                     parallelization_cores=0)

        tuner.tune(data_x=train_data, data_y=train_data_y)
        return tuner.results

    @staticmethod
    def get_ElasticRegressorHP_BayesianTuner_results():
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]

        data_y = data[target_variable]
        data_x = data.drop(columns=target_variable)

        score_list = [RmseScore()]
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=5,
            parallelization_cores=0)  # adds parallelization (per repeat)

        from hyperopt import hp
        space_lgb = {
            'alpha': hp.uniform('alpha', 0.001, 2),
            'l1_ratio': hp.uniform('l1_ratio', 0, 1),
        }

        max_evaluations = 5
        model_tuner = BayesianHyperOptModelTuner(resampler=resampler,
                                                 hyper_param_object=ElasticNetRegressorHP(),
                                                 space=space_lgb,
                                                 max_evaluations=max_evaluations,
                                                 seed=6)
        model_tuner.tune(data_x=data_x, data_y=data_y)
        return model_tuner.results

    @staticmethod
    def get_ElasticRegressorHP_GridSearch_results():
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]

        data_y = data[target_variable]
        data_x = data.drop(columns=target_variable)

        score_list = [RmseScore(), MaeScore()]

        params_dict = dict(alpha=[0.001, 0.5, 1],
                           l1_ratio=[0.001, 0.5, 1])
        grid = HyperParamsGrid(params_dict=params_dict)

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=5,
            parallelization_cores=-1)  # adds parallelization (per repeat)

        model_tuner = GridSearchModelTuner(resampler=resampler,
                                           hyper_param_object=ElasticNetRegressorHP(),
                                           params_grid=grid)
        model_tuner.tune(data_x=data_x, data_y=data_y)
        return model_tuner.results

    def test_TunerResultsComparison(self):
        light_gbm_results = TunerTests.get_LightGBMRegressor_GridSearchTuner_results()
        elastic_bayesian_results = TunerTests.get_ElasticRegressorHP_BayesianTuner_results()
        elastic_grid_results = TunerTests.get_ElasticRegressorHP_GridSearch_results()

        results_dict = {"Light GBM": light_gbm_results,
                        "Elastic (Bayesian)": elastic_bayesian_results,
                        "Elastic (Grid)": elastic_grid_results}

        comparison = TunerResultsComparison(results_dict)
        TestHelper.check_plot('data/test_Tuners/test_TunerResultsComparison_boxplot.png',
                              lambda: comparison.boxplot())

    def test_TransformationTuner(self):
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns='Survived')

        score_list = [AucRocScore(positive_class=1)]

        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)

        transformations_space = {
            # passing None will get removed, but work around is to pass a Transformer that doesn't do anything
            # i.e. EmptyTransformer
            'CenterScale vs Normalize': [EmptyTransformer(),
                                         CenterScaleTransformer(),
                                         NormalizationTransformer()],
            'PCA': [EmptyTransformer(),
                    PCATransformer()],
        }

        model_tuner = GridSearchTransformationTuner(resampler=resampler,
                                                    transformations_space=transformations_space,
                                                    hyper_param_object=LightGBMHP(),
                                                    parallelization_cores=0)
        model_tuner.tune(data_x=training_x, data_y=training_y)

        assert model_tuner.results.best_hyper_params == {'CenterScale vs Normalize': 'CenterScaleTransformer',
                                                         'PCA': 'EmptyTransformer'}
        assert model_tuner.results.best_index == 2
        assert all(model_tuner.results.sorted_best_indexes == [2, 0, 4, 3, 5, 1])
        expected_cycles = 3 * 2
        assert model_tuner.results.number_of_cycles == expected_cycles
        assert model_tuner.results.hyper_param_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.transformation_names == ['CenterScale vs Normalize', 'PCA']

        assert len(model_tuner.results.best_transformations) == 2
        assert isinstance(model_tuner.results.best_transformations[0], CenterScaleTransformer)
        assert isinstance(model_tuner.results.best_transformations[1], EmptyTransformer)

        assert not model_tuner.results.best_transformations[0].has_executed
        assert not model_tuner.results.best_transformations[1].has_executed

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        assert expected_cycles == model_tuner.results.number_of_cycles

        hyper_params_1 = transformations_space['CenterScale vs Normalize']
        hyper_params_2 = transformations_space['PCA']
        expected_transformation_combinations = list(itertools.product(hyper_params_1, hyper_params_2))
        assert len(expected_transformation_combinations) == model_tuner.results.number_of_cycles
        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)

        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        for index in range(model_tuner.results.number_of_cycles):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params,
                                                   mapping={'min_gain_to_split': 'min_split_gain',
                                                            'min_sum_hessian_in_leaf': 'min_child_weight',
                                                            'min_data_in_leaf': 'min_child_samples',
                                                            'bagging_fraction': 'subsample',
                                                            'bagging_freq': 'subsample_freq',
                                                            'feature_fraction': 'colsample_bytree',
                                                            'lambda_l1': 'reg_alpha',
                                                            'lambda_l2': 'reg_lambda'})

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 4 for original transformations + 2 hyper-paramed + 1 StatelessTransformer
                assert len(local_transformations) == 7
                assert isinstance(local_transformations[0], RemoveColumnsTransformer)
                assert isinstance(local_transformations[1], CategoricConverterTransformer)
                assert isinstance(local_transformations[2], ImputationTransformer)
                assert isinstance(local_transformations[3], DummyEncodeTransformer)

                # local_transformations has the transformations that were actually used in the Resampler
                # via the decorators
                # index_transformation_objects contains the transformations that were tuned
                index_transformation_objects = list(model_tuner.results._transformations_objects[index].values())  # noqa
                assert type(index_transformation_objects[0]) is type(local_transformations[4])  # noqa
                assert type(index_transformation_objects[1]) is type(local_transformations[5])  # noqa
                assert type(expected_transformation_combinations[index][0]) is type(local_transformations[4])  # noqa
                assert type(expected_transformation_combinations[index][1]) is type(local_transformations[5])  # noqa

                assert isinstance(local_transformations[6], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_TransformationTune_results.txt')  # noqa

        TestHelper.check_plot(
            'data/test_Tuners/test_TransformationTuner__plot_iteration_mean_scores.png',
            # noqa
            lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot(
            'data/test_Tuners/test_TransformationTuner__plot_resampled_stats.png',
            # noqa
            lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot(
            'data/test_Tuners/test_TransformationTuner__plot_resampled_scores.png',
            # noqa
            lambda: model_tuner.results.plot_resampled_scores(metric=Metric.AUC_ROC))

        assert model_tuner.results.hyper_param_names == list(transformations_space.keys())
        assert model_tuner.results.resampled_stats.shape[0] == expected_cycles

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['AUC_ROC_mean'].idxmax(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [AucRocScore(positive_class=1)]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = LightGBMHP()
        resampler.append_transformations(model_tuner.results.best_transformations)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert resampler.results.score_means['AUC_ROC'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['AUC_ROC_mean']

    def test_TransformationTuner_Regression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = []

        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns='strength')

        score_list = [RmseScore()]

        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)  # adds parallelization (per repeat)

        transformations_space = {
            # passing None will get removed, but work around is to pass a Transformer that doesn't do anything
            # i.e. EmptyTransformer
            'CenterScale vs Normalize': [EmptyTransformer(),
                                         CenterScaleTransformer(),
                                         NormalizationTransformer()],
            'PCA': [EmptyTransformer(),
                    PCATransformer()],
        }

        model_tuner = GridSearchTransformationTuner(resampler=resampler,
                                                    transformations_space=transformations_space,
                                                    hyper_param_object=LightGBMHP(),
                                                    parallelization_cores=0)
        model_tuner.tune(data_x=training_x, data_y=training_y)

        assert model_tuner.results.best_hyper_params == {'CenterScale vs Normalize': 'CenterScaleTransformer',
                                                         'PCA': 'EmptyTransformer'}
        assert model_tuner.results.best_index == 2
        assert all(model_tuner.results.sorted_best_indexes == [2, 0, 4, 5, 3, 1])
        expected_cycles = 3 * 2
        assert model_tuner.results.number_of_cycles == expected_cycles
        assert model_tuner.results.hyper_param_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.transformation_names == ['CenterScale vs Normalize', 'PCA']

        assert len(model_tuner.results.best_transformations) == 2
        assert isinstance(model_tuner.results.best_transformations[0], CenterScaleTransformer)
        assert isinstance(model_tuner.results.best_transformations[1], EmptyTransformer)

        assert not model_tuner.results.best_transformations[0].has_executed
        assert not model_tuner.results.best_transformations[1].has_executed

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        assert expected_cycles == model_tuner.results.number_of_cycles

        hyper_params_1 = transformations_space['CenterScale vs Normalize']
        hyper_params_2 = transformations_space['PCA']
        expected_transformation_combinations = list(itertools.product(hyper_params_1, hyper_params_2))
        assert len(expected_transformation_combinations) == model_tuner.results.number_of_cycles
        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)

        for index in range(len(model_tuner.resampler_decorators)):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params,
                                                   mapping={'min_gain_to_split': 'min_split_gain',
                                                            'min_sum_hessian_in_leaf': 'min_child_weight',
                                                            'min_data_in_leaf': 'min_child_samples',
                                                            'bagging_fraction': 'subsample',
                                                            'bagging_freq': 'subsample_freq',
                                                            'feature_fraction': 'colsample_bytree',
                                                            'lambda_l1': 'reg_alpha',
                                                            'lambda_l2': 'reg_lambda'},
                                                   remove_keys=['scale_pos_weight'])

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 4 for original transformations + 2 hyper-paramed + 1 StatelessTransformer
                assert len(local_transformations) == 3
                # local_transformations has the transformations that were actually used in the Resampler
                # via the decorators
                # index_transformation_objects contains the transformations that were tuned
                index_transformation_objects = list(model_tuner.results._transformations_objects[index].values())  # noqa
                assert type(index_transformation_objects[0]) is type(local_transformations[0])  # noqa
                assert type(index_transformation_objects[1]) is type(local_transformations[1])  # noqa
                assert type(expected_transformation_combinations[index][0]) is type(local_transformations[0])  # noqa
                assert type(expected_transformation_combinations[index][1]) is type(local_transformations[1])  # noqa

                assert isinstance(local_transformations[2], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_TransformationTuner_Regression_results.txt')

        TestHelper.check_plot(
            'data/test_Tuners/test_TransformationTuner_Regressionr__plot_iteration_mean_scores.png',
            lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot(
            'data/test_Tuners/test_TransformationTuner_Regressionr__plot_resampled_stats.png',
            lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot(
            'data/test_Tuners/test_TransformationTuner_Regressionr__plot_resampled_scores.png',
            lambda: model_tuner.results.plot_resampled_scores(metric=Metric.ROOT_MEAN_SQUARE_ERROR))

        assert model_tuner.results.hyper_param_names == list(transformations_space.keys())
        assert model_tuner.results.resampled_stats.shape[0] == expected_cycles

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['RMSE_mean'].idxmin(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################

        score_list = [RmseScore()]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMRegressor(),  # we'll use a Random Forest model
            transformations=[],
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = LightGBMHP()
        resampler.append_transformations(model_tuner.results.best_transformations)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert resampler.results.score_means['RMSE'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['RMSE_mean']
