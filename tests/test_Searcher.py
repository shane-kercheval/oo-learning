import itertools
import os
import shutil
from math import isclose

from oolearning import *
from tests.MockClassificationModelWrapper import MockClassificationModelWrapper
from tests.MockHyperParams import MockHyperParams
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection SpellCheckingInspection,PyUnusedLocal
class SearcherTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_modelSeracher_transformations(self):
        # we will have both global and model specific transformations. Need to make sure both work correctly
        # and independently in the Searcher.
        data = TestHelper.get_titanic_data()
        global_transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                                  CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                                  ImputationTransformer(),
                                  DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        standard_transformations = [CenterScaleTransformer(),
                                    RemoveCorrelationsTransformer()]
        score_list = [KappaScore(converter=TwoClassThresholdConverter(positive_class=1))]

        num_folds = 3
        num_repeats = 2
        ######################################################################################################
        # test that the callback is working
        ######################################################################################################
        infos = [ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=None,
                           hyper_params=None,
                           hyper_params_grid=None)]

        def train_callback(data_x, data_y, hyper_params):
            raise NotImplementedError('Test Callback')

        self.assertRaises(NotImplementedError,
                          lambda: ModelSearcher(global_transformations=[x.clone() for x in global_transformations],  # noqa
                                                model_infos=infos,
                                                splitter=ClassificationStratifiedDataSplitter(
                                                    holdout_ratio=0.25),  # noqa
                                                resampler_function=lambda m, mt: RepeatedCrossValidationResampler(  # noqa
                                                    model=m,
                                                    transformations=mt,
                                                    scores=score_list,
                                                    folds=num_folds,
                                                    repeats=num_repeats,
                                                    train_callback=train_callback)).\
                          search(data=data, target_variable='Survived'))
        ######################################################################################################
        # test Global and Model specific transformations - no global transformations, just model trans
        ######################################################################################################
        infos = [ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=[x.clone() for x in standard_transformations],
                           hyper_params=None,
                           hyper_params_grid=None)]

        # noinspection PyRedeclaration
        def train_callback(data_x, data_y, hyper_params):
            # test that the data has been center/scaled (st-dev:1, mean:0)
            assert all([isclose(x, 1) for x in data_x.std()])
            assert all([isclose(round(x, 10), 0) for x in data_x.mean()])

        ModelSearcher(global_transformations=None,
                      model_infos=infos,
                      splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                      resampler_function=lambda m, mt: RepeatedCrossValidationResampler(
                          model=m,
                          transformations=mt,
                          scores=score_list,
                          folds=num_folds,
                          repeats=num_repeats,
                          train_callback=train_callback)).search(data=data, target_variable='Survived')
        ######################################################################################################
        # test Global and Model transformations - no global transformations, just model trans
        ######################################################################################################
        infos = [ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=None,
                           hyper_params=None,
                           hyper_params_grid=None)]

        # noinspection PyRedeclaration
        def train_callback(data_x, data_y, hyper_params):
            # tests "well" enough that the transformations happen (purpose is not to test that transformations
            # are working correctly (those are different unit tests), purpose is to ensure they happen
            # note the list below is missing 'SibSp_5', because the training split is missing the uncommon
            # class after the split
            assert data_x.columns.values.tolist() == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
            # test that the data has been center/scaled (st-dev:1, mean:0)
            assert all([not isclose(x, 1) for x in data_x.std()])
            # we need to drop Parch_6 because no values (Parch == 6) will be found in some of the holdout sets
            # IN THE RESAMPLER, but the resampler will ensure that all columns exist, setting Parch_6 to all
            # 0's so the mean will be 0
            assert all([not isclose(round(x, 10), 0) for x in data_x.drop(columns='Parch_6').mean()])

        ModelSearcher(global_transformations=[x.clone() for x in global_transformations],
                      model_infos=infos,
                      splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                      resampler_function=lambda m, mt: RepeatedCrossValidationResampler(
                          model=m,
                          transformations=mt,
                          scores=score_list,
                          folds=num_folds,
                          repeats=num_repeats,
                          train_callback=train_callback)).search(data=data, target_variable='Survived')

        ######################################################################################################
        # test both Global and Model transformations
        ######################################################################################################
        infos = [ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=[x.clone() for x in standard_transformations],
                           hyper_params=None,
                           hyper_params_grid=None)]

        # noinspection PyRedeclaration
        def train_callback(data_x, data_y, hyper_params):
            # tests "well" enough that the transformations happen (purpose is not to test that transformations
            # are working correctly (those are different unit tests), purpose is to ensure they happen
            # note the list below is missing 'SibSp_5', because the training split is missing the uncommon
            # class after the split
            # NOTE: the RemoveCorrelationsTransformer will remove sex_male
            # (since it is just the opposite of sex_female)
            assert data_x.columns.values.tolist() == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa

            # test that the data has been center/scaled (st-dev:1, mean:0)
            # we need to drop Parch_6 because no values (Parch == 6) will be found in some of the holdout sets
            # IN THE RESAMPLER, but the resampler will ensure that all columns exist, setting Parch_6 to all
            # 0's so the standard deviation will be 0
            assert all([isclose(x, 1) for x in data_x.drop(columns='Parch_6').std()])
            assert all([isclose(round(x, 10), 0) for x in data_x.mean()])

        ModelSearcher(global_transformations=[x.clone() for x in global_transformations],
                      model_infos=infos,
                      splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                      resampler_function=lambda m, mt: RepeatedCrossValidationResampler(
                          model=m,
                          transformations=mt,
                          scores=score_list,
                          folds=num_folds,
                          repeats=num_repeats,
                          train_callback=train_callback)).search(data=data, target_variable='Survived')

        ######################################################################################################
        # now i want to make sure if i have multiple models, each specific model transformation will be used
        ######################################################################################################
        def raise_(ex):
            raise ex

        transformer_value_error = StatelessTransformer(custom_function=lambda x: raise_(ValueError('T1')))
        transformer_not_implemented = StatelessTransformer(custom_function=lambda x: raise_(NotImplementedError('T2')))  # noqa

        infos = [ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=[transformer_value_error]),
                 ModelInfo(description='description2',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=[x.clone() for x in standard_transformations])]
        self.assertRaises(ValueError,
                          lambda: ModelSearcher(global_transformations=[x.clone() for x in global_transformations],  # noqa
                                                model_infos=infos,
                                                splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),  # noqa
                                                resampler_function=lambda m, mt: RepeatedCrossValidationResampler(  # noqa
                                                    model=m,
                                                    transformations=mt,
                                                    scores=score_list,
                                                    folds=num_folds,
                                                    repeats=num_repeats)).search(data=data,
                                                                                 target_variable='Survived'))

        infos = [ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=[x.clone() for x in standard_transformations]),
                 ModelInfo(description='description2',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=[transformer_not_implemented])]
        self.assertRaises(NotImplementedError,
                          lambda: ModelSearcher(global_transformations=[x.clone() for x in global_transformations],  # noqa
                                                model_infos=infos,
                                                splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),  # noqa
                                                resampler_function=lambda m, mt: RepeatedCrossValidationResampler(  # noqa
                                                    model=m,
                                                    transformations=mt,
                                                    scores=score_list,
                                                    folds=num_folds,
                                                    repeats=num_repeats)).search(data=data,
                                                                                 target_variable='Survived'))

    def test_ModelSearcher_cache(self):
        data = TestHelper.get_titanic_data()
        global_transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                                  CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                                  ImputationTransformer(),
                                  DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        # Evaluators are the metrics we want to use to understand the value of our trained models.
        score_list = [KappaScore(converter=TwoClassThresholdConverter(positive_class=1)),
                      SensitivityScore(converter=TwoClassThresholdConverter(positive_class=1)),
                      SpecificityScore(converter=TwoClassThresholdConverter(positive_class=1)),
                      ErrorRateScore(converter=TwoClassThresholdConverter(positive_class=1))]
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
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=[x.clone() for x in standard_transformations],
                           hyper_params=None,
                           hyper_params_grid=None),
                 ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=None,
                           hyper_params=MockHyperParams(),
                           hyper_params_grid=grid)]

        self.assertRaises(AssertionError,
                          lambda: ModelSearcher(global_transformations=[x.clone() for x in global_transformations],  # noqa
                                                model_infos=infos,
                                                splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),  # noqa
                                                resampler_function=lambda m, mt: RepeatedCrossValidationResampler(  # noqa
                                                    model=m,
                                                    transformations=mt,
                                                    scores=score_list,
                                                    folds=num_folds,
                                                    repeats=num_repeats),
                                                persistence_manager=LocalCacheManager
                                                    (cache_directory=cache_directory)))  # noqa

        ######################################################################################################
        # test Searcher
        ######################################################################################################
        infos = [ModelInfo(description='description1',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=standard_transformations,
                           hyper_params=None,
                           hyper_params_grid=None),
                 ModelInfo(description='description2',
                           model=MockClassificationModelWrapper(data_y=data.Survived),
                           transformations=None,
                           hyper_params=MockHyperParams(),
                           hyper_params_grid=grid),
                 ModelInfo(description='dummy_stratified',
                           model=DummyClassifier(DummyClassifierStrategy.STRATIFIED),
                           transformations=None,
                           hyper_params=None,
                           hyper_params_grid=None),
                 ModelInfo(description='dummy_frequent',
                           model=DummyClassifier(DummyClassifierStrategy.MOST_FREQUENT),
                           transformations=None,
                           hyper_params=None,
                           hyper_params_grid=None)
                 ]
        model_descriptions = [x.description for x in infos]
        searcher = ModelSearcher(global_transformations=[x.clone() for x in global_transformations],
                                 model_infos=infos,
                                 splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                                 resampler_function=lambda m, mt: RepeatedCrossValidationResampler(
                                     model=m,
                                     transformations=mt,
                                     scores=score_list,
                                     folds=num_folds,
                                     repeats=num_repeats),
                                 persistence_manager=LocalCacheManager(cache_directory=cache_directory))
        searcher.search(data=data, target_variable='Survived')

        # check persistence
        # should be a file for each of the final models `final_[model description]_[hyper_params].pkl`
        assert os.path.isfile(os.path.join(cache_directory, 'final_description1_MockClassificationModelWrapper.pkl'))  # noqa
        assert os.path.isfile(os.path.join(cache_directory, 'final_description2_MockClassificationModelWrapper_criterion_gini_max_features_a_n_estimators_c_min_samples_leaf_e.pkl'))  # noqa
        assert os.path.isfile(os.path.join(cache_directory, 'final_dummy_frequent_DummyClassifier.pkl'))  # noqa
        assert os.path.isfile(os.path.join(cache_directory, 'final_dummy_stratified_DummyClassifier.pkl'))  # noqa
        # should be a directory for each of the tuned models: `tune_[model description]`
        expected_directories = [os.path.join(cache_directory, 'tune_' + x.description) for x in infos]
        assert all([os.path.isdir(x) for x in expected_directories])
        # the following models/directories should have 1 file i.e. no hyper-params to tune
        assert os.path.isfile(os.path.join(expected_directories[0], 'MockClassificationModelWrapper.pkl'))
        assert os.path.isfile(os.path.join(expected_directories[2], 'DummyClassifier.pkl'))
        assert os.path.isfile(os.path.join(expected_directories[3], 'DummyClassifier.pkl'))
        # the following model should have many hyper_params
        # build up the list of combinations of hyper_parameters and repeates/folds
        param_combinations = {'repeats': list(range(num_repeats)), 'folds': list(range(num_folds))}
        param_combinations.update(params_dict)
        params_list = [y if isinstance(y, list) else [y] for x, y in param_combinations.items()]
        expected_file_values = list(itertools.product(*params_list))
        assert len(expected_file_values) == 48
        # for each combination, ensure the file exists
        expected_file_format = os.path.join(cache_directory, 'tune_description2/repeat{0}_fold{1}_MockClassificationModelWrapper_criterion{2}_max_features{3}_n_estimators{4}_min_samples_leaf{5}.pkl')  # noqa
        assert all([os.path.isfile(expected_file_format.format(x[0], x[1], x[2], x[3], x[4], x[5])) for x in expected_file_values])  # noqa

        shutil.rmtree(cache_directory)

        # TEST TUNER RESULTS
        assert len(searcher.results.tuner_results) == 4
        assert len(searcher.results.holdout_scores) == 4

        # just one Tune result because no hyper_params
        assert len(searcher.results.tuner_results[0].tune_results) == 1
        assert len(searcher.results.tuner_results[0].time_results) == 1

        # each tuner results (grab the best model) should have num_folds * num_repeats resamples
        for index in range(len(infos)):
            assert len(searcher.results.tuner_results[index].best_model_resampler_object.cross_validation_scores) == 6  # noqa

        # same tuner should have the same results as test_resamplers_Mock_classification because of the
        # mock object, but third/fourth will be different because of dummy
        assert isclose(searcher.results.tuner_results[0].tune_results['kappa_mean'][0], -0.0024064499043792644)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['sensitivity_mean'][0], 0.371085500282958)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['specificity_mean'][0], 0.62814778309576369)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['error_rate_mean'][0], 0.47136871395048691)  # noqa

        assert isclose(searcher.results.tuner_results[0].tune_results['kappa_st_dev'][0], 0.090452035113464016)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['sensitivity_st_dev'][0], 0.061218590474225211)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['specificity_st_dev'][0], 0.031785029143322603)  # noqa
        assert isclose(searcher.results.tuner_results[0].tune_results['error_rate_st_dev'][0], 0.052884252590516621)  # noqa

        # stratified
        assert isclose(searcher.results.tuner_results[2].tune_results['kappa_mean'][0], -0.007662320952155299)  # noqa
        assert isclose(searcher.results.tuner_results[2].tune_results['sensitivity_mean'][0], 0.4099493507956116)  # noqa
        assert isclose(searcher.results.tuner_results[2].tune_results['specificity_mean'][0], 0.5827856898426659)  # noqa
        assert isclose(searcher.results.tuner_results[2].tune_results['error_rate_mean'][0], 0.48359594883324886)  # noqa

        assert isclose(searcher.results.tuner_results[2].tune_results['kappa_st_dev'][0], 0.048189576735635876)  # noqa
        assert isclose(searcher.results.tuner_results[2].tune_results['sensitivity_st_dev'][0], 0.03162499210785438)  # noqa
        assert isclose(searcher.results.tuner_results[2].tune_results['specificity_st_dev'][0], 0.01751233547463586)  # noqa
        assert isclose(searcher.results.tuner_results[2].tune_results['error_rate_st_dev'][0], 0.02628308134935678)  # noqa

        # frequent
        assert isclose(searcher.results.tuner_results[3].tune_results['kappa_mean'][0], 0)
        assert isclose(searcher.results.tuner_results[3].tune_results['sensitivity_mean'][0], 0)
        assert isclose(searcher.results.tuner_results[3].tune_results['specificity_mean'][0], 1.0)
        assert isclose(searcher.results.tuner_results[3].tune_results['error_rate_mean'][0], 0.38334207853184576)  # noqa

        assert isclose(searcher.results.tuner_results[3].tune_results['kappa_st_dev'][0], 0)
        assert isclose(searcher.results.tuner_results[3].tune_results['sensitivity_st_dev'][0], 0)
        assert isclose(searcher.results.tuner_results[3].tune_results['specificity_st_dev'][0], 0)
        assert isclose(searcher.results.tuner_results[3].tune_results['error_rate_st_dev'][0], 0.044852752350113725)  # noqa

        # should match the number of hyper-params passed in
        assert len(searcher.results.tuner_results[0].tune_results) == 1
        assert len(searcher.results.tuner_results[0].time_results) == 1
        assert len(searcher.results.tuner_results[1].tune_results) == len(grid.params_grid)
        assert len(searcher.results.tuner_results[1].time_results) == len(grid.params_grid)
        assert len(searcher.results.tuner_results[2].tune_results) == 1
        assert len(searcher.results.tuner_results[2].time_results) == 1
        assert len(searcher.results.tuner_results[3].tune_results) == 1
        assert len(searcher.results.tuner_results[3].time_results) == 1

        # same values as above (i.e. from mock ModelWrapper), but more values because we tuned across
        # many hyper-params
        assert all([isclose(x, -0.0024064499043792644) for x in searcher.results.tuner_results[1].tune_results['kappa_mean']])  # noqa
        assert all([isclose(x, 0.371085500282958) for x in searcher.results.tuner_results[1].tune_results['sensitivity_mean']])  # noqa
        assert all([isclose(x, 0.62814778309576369) for x in searcher.results.tuner_results[1].tune_results['specificity_mean']])  # noqa
        assert all([isclose(x, 0.47136871395048691) for x in searcher.results.tuner_results[1].tune_results['error_rate_mean']])  # noqa

        assert all([isclose(x, 0.090452035113464016) for x in searcher.results.tuner_results[1].tune_results['kappa_st_dev']])  # noqa
        assert all([isclose(x, 0.061218590474225211) for x in searcher.results.tuner_results[1].tune_results['sensitivity_st_dev']])  # noqa
        assert all([isclose(x, 0.031785029143322603) for x in searcher.results.tuner_results[1].tune_results['specificity_st_dev']])  # noqa
        assert all([isclose(x, 0.052884252590516621) for x in searcher.results.tuner_results[1].tune_results['error_rate_st_dev']])  # noqa

        # TEST HOLDOUT SCORES
        assert len(searcher.results.holdout_scores) == 4  # 4 models
        assert len(searcher.results.holdout_scores[0]) == 4  # 4 Evaluators
        assert len(searcher.results.holdout_scores[1]) == 4  # 4 Evaluators
        assert len(searcher.results.holdout_scores[2]) == 4  # 4 Evaluators
        assert len(searcher.results.holdout_scores[3]) == 4  # 4 Evaluators
        assert all([x == y for x, y in zip([x.name for x in searcher.results.holdout_scores[0]], ['kappa', 'sensitivity', 'specificity', 'error_rate'])])  # noqa
        assert all([x == y for x, y in zip([x.name for x in searcher.results.holdout_scores[1]], ['kappa', 'sensitivity', 'specificity', 'error_rate'])])  # noqa
        assert all([x == y for x, y in zip([x.name for x in searcher.results.holdout_scores[2]], ['kappa', 'sensitivity', 'specificity', 'error_rate'])])  # noqa
        assert all([x == y for x, y in zip([x.name for x in searcher.results.holdout_scores[3]], ['kappa', 'sensitivity', 'specificity', 'error_rate'])])  # noqa

        assert all([isclose(x, y) for x, y in zip([x.value for x in searcher.results.holdout_scores[0]], [0.02628424657534245, 0.38372093023255816, 0.6423357664233577, 0.45739910313901344])])  # noqa
        assert all([isclose(x, y) for x, y in zip([x.value for x in searcher.results.holdout_scores[1]], [0.02628424657534245, 0.38372093023255816, 0.6423357664233577, 0.45739910313901344])])  # noqa
        # same values that are in `fitter.training_evaluator.all_quality_metrics` in test_ModelWrappers
        assert all([isclose(x, y) for x, y in zip([x.value for x in searcher.results.holdout_scores[2]], [0.10655528087972044, 0.4418604651162791, 0.6642335766423357, 0.42152466367713004])])  # noqa
        assert all([isclose(x, y) for x, y in zip([x.value for x in searcher.results.holdout_scores[3]], [0.0, 0.0, 1.0, 0.38565022421524664])])  # noqa
        # same values for indexes 2,3 (DummyClassifiers) that are in
        # `fitter.training_evaluator.all_quality_metrics` in test_ModelWrappers
        assert all(searcher.results.holdout_score_values.index.values == model_descriptions)
        assert all(searcher.results.holdout_score_values.columns.values == ['kappa', 'sensitivity', 'specificity', 'error_rate'])
        assert all([isclose(x, y) for x, y in zip(list(searcher.results.holdout_score_values.kappa), [0.02628424657534245, 0.02628424657534245, 0.10655528087972044, 0.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(list(searcher.results.holdout_score_values.sensitivity), [0.38372093023255816, 0.38372093023255816, 0.4418604651162791, 0.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(list(searcher.results.holdout_score_values.specificity), [0.6423357664233577, 0.6423357664233577, 0.6642335766423357, 1.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(list(searcher.results.holdout_score_values.error_rate), [0.45739910313901344, 0.45739910313901344, 0.42152466367713004, 0.38565022421524664])])  # noqa

        values = list(searcher.results.holdout_score_values.kappa)
        highest_kappa = values.index(max(values))
        assert searcher.results.best_model_index == highest_kappa

        assert all(searcher.results.best_tuned_results.index.values == model_descriptions)
        assert all(searcher.results.best_tuned_results.model == ['MockClassificationModelWrapper', 'MockClassificationModelWrapper', 'DummyClassifier', 'DummyClassifier'])  # noqa
        assert all(searcher.results.best_tuned_results.hyper_params == [{'hyper_params': 'None'}, {'criterion': 'gini', 'max_features': 'a', 'n_estimators': 'c', 'min_samples_leaf': 'e'}, {'hyper_params': 'None'}, {'hyper_params': 'None'}])  # noqa

        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.kappa_mean, [-0.0024064499043793632, -0.0024064499043793632, -0.007662320952155299, 0.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.kappa_st_dev, [0.090452035113464016, 0.090452035113464016])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.fillna(0).kappa_cv, [-37.59, -37.59, -6.29, 0])])  # noqa

        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.sensitivity_mean, [0.371085500282958, 0.371085500282958, 0.4099493507956116, 0.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.sensitivity_st_dev, [0.06121859047422521, 0.06121859047422521, 0.03162499210785438, 0.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.fillna(0).sensitivity_cv, [0.16, 0.16, 0.08, 0])])  # noqa

        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.specificity_mean, [0.6281477830957637, 0.6281477830957637, 0.5827856898426659, 1.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.specificity_st_dev, [0.0317850291433226, 0.0317850291433226, 0.01751233547463586, 0.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.specificity_cv, [0.05, 0.05, 0.03, 0.0])])  # noqa

        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.error_rate_mean, [0.4713687139504869, 0.4713687139504869, 0.48359594883324886, 0.38334207853184576])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.error_rate_st_dev, [0.05288425259051662, 0.05288425259051662, 0.02628308134935678, 0.044852752350113725])])  # noqa
        assert all([isclose(x, y) for x, y in zip(searcher.results.best_tuned_results.error_rate_cv, [0.11, 0.11, 0.05, 0.12])])  # noqa

        TestHelper.check_plot('data/test_Searcher/test_get_holdout_score_heatmap.png',
                              lambda: searcher.results.get_holdout_score_heatmap())
        TestHelper.check_plot('data/test_Searcher/test_get_resamples_boxplot_KAPPA.png',
                              lambda: searcher.results.get_resamples_boxplot(Metric.KAPPA))
        TestHelper.check_plot('data/test_Searcher/test_get_resamples_boxplot_SENSITIVITY.png',
                              lambda: searcher.results.get_resamples_boxplot(Metric.SENSITIVITY))
        TestHelper.check_plot('data/test_Searcher/test_get_resamples_boxplot_SPECIFICITY.png',
                              lambda: searcher.results.get_resamples_boxplot(Metric.SPECIFICITY))
        TestHelper.check_plot('data/test_Searcher/test_get_resamples_boxplot_ERROR_RATE.png',
                              lambda: searcher.results.get_resamples_boxplot(Metric.ERROR_RATE))
