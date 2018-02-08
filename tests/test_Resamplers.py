import os
import pickle
from math import isclose

import numpy as np
import matplotlib.pyplot as plt

from oolearning import *

from tests.MockClassificationModelWrapper import MockClassificationModelWrapper
from tests.MockRegressionModelWrapper import MockRegressionModelWrapper
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection SpellCheckingInspection,PyMethodMayBeStatic
class ResamplerTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_resamplers_Rmse_Mae(self):
        data = TestHelper.get_cement_data()
        # splitter = RegressionStratifiedDataSplitter(test_ratio=0.2)
        # training_indexes, test_indexes = splitter.split(target_values=data.strength)

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop(columns='strength')

        # test_data = data.iloc[test_indexes]
        # test_data_y = test_data.strength
        # test_data = test_data.drop(columns='strength')

        resampler = RepeatedCrossValidationResampler(
            model=RegressionMW(),
            model_transformations=ModelDefaults.transformations_regression(),
            evaluators=[RmseEvaluator(),
                        MaeEvaluator()],
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y)
        assert len(resampler.results._evaluators) == 25
        assert all([len(x) == 2 and
                    isinstance(x[0], RmseEvaluator) and
                    isinstance(x[1], MaeEvaluator)
                    for x in resampler.results._evaluators])
        assert resampler.results.num_resamples == 25
        assert resampler.results.metrics == ['RMSE', 'MAE']
        assert isclose(resampler.results.metric_means['RMSE'], 10.459344010622544)
        assert isclose(resampler.results.metric_means['MAE'], 8.2855537849498742)
        assert isclose(resampler.results.metric_standard_deviations['RMSE'], 0.5716680069548794)
        assert isclose(resampler.results.metric_standard_deviations['MAE'], 0.46714447004190812)
        assert isclose(resampler.results.metric_coefficient_of_variation['RMSE'], round(0.5716680069548794 / 10.459344010622544, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['MAE'], round(0.46714447004190812 / 8.2855537849498742, 2))  # noqa

        actual_cross_validations = resampler.results.cross_validation_scores
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Resamplers/test_resamplers_Rmse_Mae_cross_validation_scores.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(actual_cross_validations, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_cross_validations = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_cross_validations,
                                                      data_frame2=actual_cross_validations)

    def test_resamplers_Mock_regression(self):
        data = TestHelper.get_cement_data()
        # splitter = RegressionStratifiedDataSplitter(test_ratio=0.2)
        # training_indexes, test_indexes = splitter.split(target_values=data.strength)

        train_data = data
        train_data_y = train_data.strength
        train_data = train_data.drop(columns='strength')

        # test_data = data.iloc[test_indexes]
        # test_data_y = test_data.strength
        # test_data = test_data.drop(columns='strength')

        resampler = RepeatedCrossValidationResampler(
            model=MockRegressionModelWrapper(data_y=data.strength),
            model_transformations=ModelDefaults.transformations_regression(),
            evaluators=[RmseEvaluator(),
                        MaeEvaluator()],
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y)
        assert len(resampler.results._evaluators) == 25
        assert all([len(x) == 2 and
                    isinstance(x[0], RmseEvaluator) and
                    isinstance(x[1], MaeEvaluator)
                    for x in resampler.results._evaluators])
        assert resampler.results.num_resamples == 25
        assert resampler.results.metrics == ['RMSE', 'MAE']
        assert isclose(resampler.results.metric_means['RMSE'], 23.776598887994158)
        assert isclose(resampler.results.metric_means['MAE'], 19.030724889732316)
        assert isclose(resampler.results.metric_standard_deviations['RMSE'], 0.91016288102942078)
        assert isclose(resampler.results.metric_standard_deviations['MAE'], 0.77294039453317798)
        assert isclose(resampler.results.metric_coefficient_of_variation['RMSE'], round(0.91016288102942078 / 23.776598887994158, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['MAE'], round(0.77294039453317798 / 19.030724889732316, 2))  # noqa

    def test_resamplers_Mock_classification(self):
        data = TestHelper.get_titanic_data()

        # main reason we want to split the data is to get the means/st_devs so that we can confirm with
        # e.g. the Searcher
        splitter = ClassificationStratifiedDataSplitter(test_ratio=0.25)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.Survived
        train_data = train_data.drop(columns='Survived')

        evaluator_list = [KappaEvaluator(positive_category=1,
                                         negative_category=0,
                                         threshold=0.5),
                          SensitivityEvaluator(positive_category=1,
                                               negative_category=0,
                                               threshold=0.5),
                          SpecificityEvaluator(positive_category=1,
                                               negative_category=0,
                                               threshold=0.5),
                          ErrorRateTwoClassEvaluator(positive_category=1,
                                                     negative_category=0,
                                                     threshold=0.5)]

        resampler = RepeatedCrossValidationResampler(
            model=MockClassificationModelWrapper(data_y=data.Survived),
            model_transformations=ModelDefaults.transformations_random_forest(),
            evaluators=evaluator_list,
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y)
        assert len(resampler.results._evaluators) == 25
        assert all([len(x) == 4 and
                    isinstance(x[0], KappaEvaluator) and
                    isinstance(x[1], SensitivityEvaluator) and
                    isinstance(x[2], SpecificityEvaluator) and
                    isinstance(x[3], ErrorRateTwoClassEvaluator)
                    for x in resampler.results._evaluators])
        assert resampler.results.num_resamples == 25
        assert resampler.results.metrics == ['kappa', 'sensitivity', 'specificity', 'ErrorRate']
        assert isclose(resampler.results.metric_means['kappa'], 0.0013793651663756446)
        assert isclose(resampler.results.metric_means['sensitivity'], 0.34802926509722726)
        assert isclose(resampler.results.metric_means['specificity'], 0.65307336918498493)
        assert isclose(resampler.results.metric_means['ErrorRate'], 0.46314142734094416)

        assert isclose(resampler.results.metric_standard_deviations['kappa'], 0.055624736458973652)
        assert isclose(resampler.results.metric_standard_deviations['sensitivity'], 0.036787308260115267)
        assert isclose(resampler.results.metric_standard_deviations['specificity'], 0.019357626459983342)
        assert isclose(resampler.results.metric_standard_deviations['ErrorRate'], 0.025427045943705647)

        assert isclose(resampler.results.metric_coefficient_of_variation['kappa'], round(0.055624736458973652 / 0.0013793651663756446, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['sensitivity'], round(0.036787308260115267 / 0.34802926509722726, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['specificity'], round(0.019357626459983342 / 0.65307336918498493, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['ErrorRate'], round(0.025427045943705647 / 0.46314142734094416, 2))  # noqa

    def test_Resampler_callback(self):
        # make sure that the Resampler->train_callback works
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        # noinspection PyUnusedLocal
        def train_callback(data_x, data_y, hyper_params):
            raise NotImplementedError()

        evaluators = [RmseEvaluator(), MaeEvaluator()]
        transformations = [RemoveColumnsTransformer(['coarseagg', 'fineagg']), ImputationTransformer(), DummyEncodeTransformer()]  # noqa
        resampler = RepeatedCrossValidationResampler(
            model=RandomForestMW(),
            model_transformations=transformations,
            evaluators=evaluators,
            folds=5,
            repeats=5,
            train_callback=train_callback)

        # should raise an error from the callback definition above
        self.assertRaises(NotImplementedError, lambda: resampler.resample(data_x=data.drop(columns=target_variable), data_y=data[target_variable], hyper_params=None))  # noqa

    def test_Resampler_transformations(self):
        # intent of this test is to ensure that the data is being transformed according to the
        # transformations being passed in.

        # make sure that the Resampler->train_callback works
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        # create random missing values and extra field
        np.random.seed(42)
        missing_indexes_cement = np.random.randint(low=0, high=len(data), size=int(len(data) * 0.10))
        data.loc[missing_indexes_cement, 'cement'] = None

        np.random.seed(43)
        missing_indexes_ash = np.random.randint(low=0, high=len(data), size=int(len(data) * 0.10))
        data.loc[missing_indexes_ash, 'ash'] = None

        np.random.seed(42)
        random_codes = np.random.randint(low=0, high=2, size=len(data))
        data['random'] = ['code0' if random_code == 0 else 'code1' for random_code in random_codes]

        assert data.isna().sum().sum() == 195

        data_x = data.drop(columns=target_variable)
        data_y = data[target_variable]

        ######################################################################################################
        # make sure the data that we pass to `train()` in the ModelWrapper is transformed
        # then make sure what we get in the callback matches the transformed data
        ######################################################################################################
        test_pipeline = TransformerPipeline(
            transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),  # noqa
                             ImputationTransformer(),
                             DummyEncodeTransformer()])
        transformed_data = test_pipeline.fit_transform(data_x=data_x)
        # make sure our test transformations are transformed as expected (although this should already be
        # tested in test_Transformations file
        assert all(transformed_data.columns.values == ['cement', 'slag', 'ash', 'water', 'superplastic', 'age', 'random_code1'])  # noqa
        assert OOLearningHelpers.is_series_numeric(variable=transformed_data.random_code1)
        assert transformed_data.isna().sum().sum() == 0

        # this callback will be called by the ModelWrapper before fitting the model
        # the callback gives us back the data that it will pass to the underlying model
        # so we can make sure it matches what we expect
        def train_callback(data_x_test, data_y_test, hyper_params):
            assert hyper_params is None
            # noinspection PyTypeChecker
            assert all(data_y == data_y_test)
            TestHelper.ensure_all_values_equal(data_frame1=transformed_data, data_frame2=data_x_test)

        evaluators = [RmseEvaluator(), MaeEvaluator()]
        transformations = [RemoveColumnsTransformer(['coarseagg', 'fineagg']), ImputationTransformer(), DummyEncodeTransformer()]  # noqa
        resampler = RepeatedCrossValidationResampler(
            model=MockRegressionModelWrapper(data_y=data_y),
            model_transformations=transformations,
            evaluators=evaluators,
            folds=5,
            repeats=5,
            train_callback=train_callback)

        # the train_callback method will be triggered and will cause an assertion error if the data that is
        # going to be trained does not match the data previously transformed
        resampler.resample(data_x=data.drop(columns=target_variable), data_y=data[target_variable], hyper_params=None)  # noqa

    def test_resamplers_RandomForest_classification(self):
        data = TestHelper.get_titanic_data()

        # main reason we want to split the data is to get the means/st_devs so that we can confirm with
        # e.g. the Searcher
        splitter = ClassificationStratifiedDataSplitter(test_ratio=0.25)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.Survived
        train_data = train_data.drop(columns='Survived')

        # noinspection SpellCheckingInspection
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        evaluator_list = [KappaEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          SensitivityEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          SpecificityEvaluator(positive_category=1, negative_category=0, threshold=0.5),
                          ErrorRateTwoClassEvaluator(positive_category=1, negative_category=0, threshold=0.5)]

        cache_directory = TestHelper.ensure_test_directory('data/test_Resamplers/cached_test_models/test_resamplers_RandomForest_classification')  # noqa
        resampler = RepeatedCrossValidationResampler(
            model=RandomForestMW(),
            model_transformations=transformations,
            evaluators=evaluator_list,
            persistence_manager=LocalCacheManager(cache_directory=cache_directory),
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP())
        assert len(resampler.results._evaluators) == 25
        assert all([len(x) == 4 and
                    isinstance(x[0], KappaEvaluator) and
                    isinstance(x[1], SensitivityEvaluator) and
                    isinstance(x[2], SpecificityEvaluator) and
                    isinstance(x[3], ErrorRateTwoClassEvaluator)
                    for x in resampler.results._evaluators])
        assert resampler.results.num_resamples == 25

        # noinspection SpellCheckingInspection
        expected_file = 'repeat{0}_fold{1}_RandomForestMW_n_estimators500_criteriongini_max_featuresNone_max_depthNone_min_samples_split2_min_samples_leaf1_min_weight_fraction_leaf0.0_max_leaf_nodesNone_min_impurity_decrease0_bootstrapTrue_oob_scoreFalse_n_jobs-1_random_state42.pkl'  # noqa
        for fold_index in range(5):
            for repeat_index in range(5):
                assert os.path.isfile(os.path.join(cache_directory,
                                                   expected_file.format(fold_index, repeat_index)))

        assert resampler.results.metrics == ['kappa', 'sensitivity', 'specificity', 'ErrorRate']

        # make sure the order of the cross_validation_scores is the same order as Evaluators passed in
        assert all(resampler.results.cross_validation_scores.columns.values == ['kappa', 'sensitivity', 'specificity', 'ErrorRate'])  # noqa

        # metric_means and metric_standard_deviations comes from cross_validation_scores, so testing both
        assert isclose(resampler.results.metric_means['kappa'], 0.5871647457625514)
        assert isclose(resampler.results.metric_means['sensitivity'], 0.7225285066820597)
        assert isclose(resampler.results.metric_means['specificity'], 0.86179768930490386)
        assert isclose(resampler.results.metric_means['ErrorRate'], 0.19173962767419936)

        assert isclose(resampler.results.metric_standard_deviations['kappa'], 0.065991891925931981)
        assert isclose(resampler.results.metric_standard_deviations['sensitivity'], 0.065086179854148496)
        assert isclose(resampler.results.metric_standard_deviations['specificity'], 0.035803688812849649)
        assert isclose(resampler.results.metric_standard_deviations['ErrorRate'], 0.029873757630591153)

        assert isclose(resampler.results.metric_coefficient_of_variation['kappa'], round(0.065991891925931981 / 0.5871647457625514, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['sensitivity'], round(0.065086179854148496 / 0.7225285066820597, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['specificity'], round(0.035803688812849649 / 0.86179768930490386, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['ErrorRate'], round(0.029873757630591153 / 0.19173962767419936, 2))  # noqa

        plt.gcf().clear()
        TestHelper.check_plot('data/test_Resamplers/test_resamplers_RandomForest_classification_cv_boxplot.png',  # noqa
                              lambda: resampler.results.cross_validation_score_boxplot())
