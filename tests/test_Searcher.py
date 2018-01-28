import itertools
import os
import shutil
from math import isclose

from oolearning import *

from tests.MockClassificationModelWrapper import MockClassificationModelWrapper
from tests.MockHyperParams import MockHyperParams
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection SpellCheckingInspection
class TunerTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_ModelSearcher_cache(self):
        data = TestHelper.get_titanic_data()
        global_transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                                  CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                                  ImputationTransformer(),
                                  DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        # Evaluators are the metrics we want to use to understand the value of our trained models.
        evaluator_list = [KappaEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          SensitivityEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          SpecificityEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          ErrorRateTwoClassEvaluator(positive_category=1, negative_category=0, threshold=0.5)]
        standard_transformations = [CenterScaleTransformer(),
                                    RemoveCorrelationsTransformer()]

        params_dict = {'criterion': 'gini', 'max_features': ['a', 'b'], 'n_estimators': ['c', 'd'], 'min_samples_leaf': ['e', 'f']}  # noqa
        grid = HyperParamsGrid(params_dict=params_dict)

        cache_directory = TestHelper.ensure_test_directory('data/test_Searcher/cached_test_models/test_ModelSearcher_cache')  # noqa
        assert os.path.isdir(cache_directory) is False

        num_folds = 3
        num_repeats = 2
        ######################################################################################################
        # Searcher must have unique descriptions
        ######################################################################################################
        infos = [ModelInfo(description='description1',
                           model_wrapper=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=standard_transformations,
                           hyper_params=None,
                           hyper_params_grid=None),
                 ModelInfo(description='description1',
                           model_wrapper=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=None,
                           hyper_params=MockHyperParams(),
                           hyper_params_grid=grid)]

        self.assertRaises(AssertionError,
                          lambda: ModelSearcher(global_transformations=global_transformations,
                                                model_infos=infos,
                                                splitter=ClassificationStratifiedDataSplitter(test_ratio=0.25),  # noqa
                                                resampler_function=lambda m, mt: RepeatedCrossValidationResampler(  # noqa
                                                    model=m,
                                                    model_transformations=mt,
                                                    evaluators=evaluator_list,
                                                    folds=num_folds,
                                                    repeats=num_repeats),
                                                persistence_manager=LocalCacheManager
                                                    (cache_directory=cache_directory)))  # noqa
        ######################################################################################################
        # test Searcher
        ######################################################################################################
        infos = [ModelInfo(description='description1',
                           model_wrapper=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=standard_transformations,
                           hyper_params=None,
                           hyper_params_grid=None),
                 ModelInfo(description='description2',
                           model_wrapper=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=None,
                           hyper_params=MockHyperParams(),
                           hyper_params_grid=grid)]
        model_descriptions = [x.description for x in infos]
        searcher = ModelSearcher(global_transformations=global_transformations,
                                 model_infos=infos,
                                 splitter=ClassificationStratifiedDataSplitter(test_ratio=0.25),
                                 resampler_function=lambda m, mt: RepeatedCrossValidationResampler(
                                     model=m,
                                     model_transformations=mt,
                                     evaluators=evaluator_list,
                                     folds=num_folds,
                                     repeats=num_repeats),
                                 persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        searcher.search(data_x=data.drop('Survived', axis=1), data_y=data.Survived)

        # check persistence
        # first check cache files for tuning
        assert os.path.isfile(os.path.join(cache_directory, 'tune_description1_MockClassificationModelWrapper.pkl'))  # noqa
        # build up the list of combinations of hyper_parameters and repeates/folds
        param_combinations = {'repeats': list(range(num_repeats)), 'folds': list(range(num_folds))}
        param_combinations.update(params_dict)
        params_list = [y if isinstance(y, list) else [y] for x, y in param_combinations.items()]
        expected_file_values = list(itertools.product(*params_list))
        assert len(expected_file_values) == 48
        # for each combination, ensure the file exists
        expected_file_format = os.path.join(cache_directory, 'tune_description2_repeat{0}_fold{1}_MockClassificationModelWrapper_criterion{2}_max_features{3}_n_estimators{4}_min_samples_leaf{5}.pkl')  # noqa
        assert [os.path.isfile(expected_file_format.format(x[0], x[1], x[2], x[3], x[4], x[5])) for x in expected_file_values]  # noqa

        # now check cache files for the holdout set
        assert os.path.isfile(os.path.join(cache_directory, 'holdout_description1_MockClassificationModelWrapper.pkl'))  # noqa
        assert os.path.isfile(os.path.join(cache_directory, 'holdout_description2_MockClassificationModelWrapper_criterion_gini_max_features_a_n_estimators_c_min_samples_leaf_e.pkl'))  # noqa

        shutil.rmtree(cache_directory)

        assert len(searcher.results.tuner_results) == 2
        assert len(searcher.results.holdout_evaluators) == 2

        # just one Tune result because no hyper_params
        assert len(searcher.results.tuner_results[0].tune_results) == 1
        assert len(searcher.results.tuner_results[0].time_results) == 1

        # same tuner should have the same results as test_resamplers_Mock_classification because of the
        # mock object
        assert isclose(searcher.results.tuner_results[0].tune_results['kappa_mean'][0], -0.0024064499043792644)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['sensitivity_mean'][0], 0.371085500282958)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['specificity_mean'][0], 0.62814778309576369)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['ErrorRate_mean'][0], 0.47136871395048691)  # noqa

        assert isclose(searcher.results.tuner_results[0].tune_results['kappa_st_dev'][0], 0.090452035113464016)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['sensitivity_st_dev'][0], 0.061218590474225211)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['specificity_st_dev'][0], 0.031785029143322603)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['ErrorRate_st_dev'][0], 0.052884252590516621)  # noqa

        # should match the number of hyper-params passed in
        assert len(searcher.results.tuner_results[1].tune_results) == len(grid.params_grid)
        assert len(searcher.results.tuner_results[1].time_results) == len(grid.params_grid)

        # same values as above (i.e. from mock ModelWrapper), but more values because we tuned across
        # many hyper-params
        assert all([isclose(x, -0.0024064499043792644) for x in searcher.results.tuner_results[1].tune_results['kappa_mean']])  # noqa
        assert all([isclose(x, 0.371085500282958) for x in searcher.results.tuner_results[1].tune_results['sensitivity_mean']])  # noqa
        assert all([isclose(x, 0.62814778309576369) for x in searcher.results.tuner_results[1].tune_results['specificity_mean']])  # noqa
        assert all([isclose(x, 0.47136871395048691) for x in searcher.results.tuner_results[1].tune_results['ErrorRate_mean']])  # noqa

        assert all([isclose(x, 0.090452035113464016) for x in searcher.results.tuner_results[1].tune_results['kappa_st_dev']])  # noqa
        assert all([isclose(x, 0.061218590474225211) for x in searcher.results.tuner_results[1].tune_results['sensitivity_st_dev']])  # noqa
        assert all([isclose(x, 0.031785029143322603) for x in searcher.results.tuner_results[1].tune_results['specificity_st_dev']])  # noqa
        assert all([isclose(x, 0.052884252590516621) for x in searcher.results.tuner_results[1].tune_results['ErrorRate_st_dev']])  # noqa

        assert len(searcher.results.holdout_evaluators) == 2  # 2 models
        assert len(searcher.results.holdout_evaluators[0]) == 4  # 4 Evaluators
        assert len(searcher.results.holdout_evaluators[1]) == 4  # 4 Evaluators
        assert [x.metric_name for x in searcher.results.holdout_evaluators[0]] == ['kappa', 'sensitivity', 'specificity', 'ErrorRate']  # noqa
        assert [x.metric_name for x in searcher.results.holdout_evaluators[1]] == ['kappa', 'sensitivity', 'specificity', 'ErrorRate']  # noqa
        assert [x.value for x in searcher.results.holdout_evaluators[0]] == [0.026284246575342427, 0.38372093023255816, 0.64233576642335766, 0.45739910313901344]  # noqa

        assert len(searcher.results.holdout_evaluators) == 2  # 2 models
        assert len(searcher.results.holdout_evaluators[1]) == 4  # 4 Evaluators
        assert [x.value for x in searcher.results.holdout_evaluators[1]] == [0.026284246575342427, 0.38372093023255816, 0.64233576642335766, 0.45739910313901344]  # noqa

        assert all(searcher.results.holdout_eval_values.index.values == model_descriptions)
        assert all(searcher.results.holdout_eval_values.columns.values == ['kappa', 'sensitivity', 'specificity', 'ErrorRate'])  # noqa
        assert all([isclose(x, 0.026284246575342427) for x in searcher.results.holdout_eval_values.kappa])
        assert all([isclose(x, 0.38372093023255816) for x in searcher.results.holdout_eval_values.sensitivity])  # noqa
        assert all([isclose(x, 0.64233576642335766) for x in searcher.results.holdout_eval_values.specificity])  # noqa
        assert all([isclose(x, 0.45739910313901344) for x in searcher.results.holdout_eval_values.ErrorRate])

        # The mock model wrapper will return the same accuracies for each model so the first model will be
        # selected as the best
        assert searcher.results.best_model_index == 0

        assert all(searcher.results.best_tuned_results.index.values == model_descriptions)
        assert all(searcher.results.best_tuned_results.model == ['MockClassificationModelWrapper', 'MockClassificationModelWrapper'])  # noqa
        assert all(searcher.results.best_tuned_results.hyper_params == [{'hyper_params': 'None'}, {'criterion': 'gini', 'max_features': 'a', 'n_estimators': 'c', 'min_samples_leaf': 'e'}])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.kappa_mean, [-0.0024064499043792644, -0.0024064499043792644])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.kappa_st_dev, [0.090452035113464016, 0.090452035113464016])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.sensitivity_mean, [0.371085500282958, 0.371085500282958])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.sensitivity_st_dev, [0.061218590474225211, 0.061218590474225211])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.specificity_mean, [0.62814778309576369, 0.62814778309576369])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.specificity_st_dev, [0.031785029143322603, 0.031785029143322603])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.ErrorRate_mean, [0.47136871395048691, 0.47136871395048691])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.ErrorRate_st_dev, [0.052884252590516621, 0.052884252590516621])])  # noqa

        # searcher.search_results.best_model
        # searcher.get_heatmap()
        # searcher.get_graph()
