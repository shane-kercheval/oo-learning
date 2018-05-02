import os
import pickle
from math import isclose

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
            model=LinearRegressor(),
            transformations=[ImputationTransformer(),
                             DummyEncodeTransformer(CategoricalEncoding.DUMMY)],
            scores=[RmseScore(),
                    MaeScore()],
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y)
        assert len(resampler.results._scores) == 25
        assert all([len(x) == 2 and
                    isinstance(x[0], RmseScore) and
                    isinstance(x[1], MaeScore)
                    for x in resampler.results._scores])
        assert resampler.results.num_resamples == 25
        assert resampler.results.metrics == ['RMSE', 'MAE']
        assert isclose(resampler.results.metric_means['RMSE'], 10.459344010622544)
        assert isclose(resampler.results.metric_means['MAE'], 8.2855537849498742)
        assert isclose(resampler.results.metric_standard_deviations['RMSE'], 0.5716680069548794)
        assert isclose(resampler.results.metric_standard_deviations['MAE'], 0.46714447004190812)
        assert isclose(resampler.results.metric_coefficient_of_variation['RMSE'], round(0.5716680069548794 / 10.459344010622544, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['MAE'], round(0.46714447004190812 / 8.2855537849498742, 2))  # noqa

        actual_cross_validations = resampler.results.resampled_scores
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
            transformations=[ImputationTransformer(),
                             DummyEncodeTransformer(CategoricalEncoding.DUMMY)],
            scores=[RmseScore(),
                    MaeScore()],
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y)
        assert len(resampler.results._scores) == 25
        assert all([len(x) == 2 and
                    isinstance(x[0], RmseScore) and
                    isinstance(x[1], MaeScore)
                    for x in resampler.results._scores])
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
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.25)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.Survived
        train_data = train_data.drop(columns='Survived')

        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
                          SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa

        resampler = RepeatedCrossValidationResampler(
            model=MockClassificationModelWrapper(data_y=data.Survived),
            transformations=None,
            scores=score_list,
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y)
        assert len(resampler.results._scores) == 25
        assert all([len(x) == 4 and
                    isinstance(x[0], KappaScore) and
                    isinstance(x[1], SensitivityScore) and
                    isinstance(x[2], SpecificityScore) and
                    isinstance(x[3], ErrorRateScore)
                    for x in resampler.results._scores])
        assert resampler.results.num_resamples == 25
        assert resampler.results.metrics == ['kappa', 'sensitivity', 'specificity', 'error_rate']
        assert isclose(resampler.results.metric_means['kappa'], 0.0013793651663756446)
        assert isclose(resampler.results.metric_means['sensitivity'], 0.34802926509722726)
        assert isclose(resampler.results.metric_means['specificity'], 0.65307336918498493)
        assert isclose(resampler.results.metric_means['error_rate'], 0.46314142734094416)

        assert isclose(resampler.results.metric_standard_deviations['kappa'], 0.055624736458973652)
        assert isclose(resampler.results.metric_standard_deviations['sensitivity'], 0.036787308260115267)
        assert isclose(resampler.results.metric_standard_deviations['specificity'], 0.019357626459983342)
        assert isclose(resampler.results.metric_standard_deviations['error_rate'], 0.025427045943705647)

        assert isclose(resampler.results.metric_coefficient_of_variation['kappa'], round(0.055624736458973652 / 0.0013793651663756446, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['sensitivity'], round(0.036787308260115267 / 0.34802926509722726, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['specificity'], round(0.019357626459983342 / 0.65307336918498493, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['error_rate'], round(0.025427045943705647 / 0.46314142734094416, 2))  # noqa

    def test_Resampler_callback(self):
        # make sure that the Resampler->train_callback works
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        # noinspection PyUnusedLocal
        def train_callback(data_x, data_y, hyper_params):
            raise NotImplementedError()

        score_list = [RmseScore(), MaeScore()]
        transformations = [RemoveColumnsTransformer(['coarseagg', 'fineagg']), ImputationTransformer(), DummyEncodeTransformer()]  # noqa
        resampler = RepeatedCrossValidationResampler(
            model=RandomForestClassifier(),
            transformations=transformations,
            scores=score_list,
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
            # make sure transformations happened
            assert all(data_x_test.columns.values == ['cement', 'slag', 'ash', 'water', 'superplastic', 'age', 'random_code1'])  # noqa

        score_list = [RmseScore(), MaeScore()]
        transformations = [RemoveColumnsTransformer(['coarseagg', 'fineagg']), ImputationTransformer(), DummyEncodeTransformer()]  # noqa
        resampler = RepeatedCrossValidationResampler(
            model=MockRegressionModelWrapper(data_y=data_y),
            transformations=transformations,
            scores=score_list,
            folds=5,
            repeats=5,
            train_callback=train_callback)

        # the train_callback method will be triggered and will cause an assertion error if the data that is
        # going to be trained does not match the data previously transformed
        resampler.resample(data_x=data.drop(columns=target_variable), data_y=data[target_variable], hyper_params=None)  # noqa

    def test_Resampler_fold_indexes(self):
        # test that the resampler uses the same fold index across objects. Test that the indexes are
        # maintained in the predicted values (only applicable for dataframes i.e. classification)
        class TempDecorator(DecoratorBase):
            def __init__(self):
                self._repeat_index = list()
                self._fold_index = list()
                self._holdout_indexes = list()
                self._holdout_predicted_values = pd.DataFrame()

            def decorate(self, **kwargs):
                self._repeat_index.append(kwargs['repeat_index'])
                self._fold_index.append(kwargs['fold_index'])
                self._holdout_indexes.extend(kwargs['holdout_indexes'])
                self._holdout_predicted_values = self._holdout_predicted_values.append(kwargs['holdout_predicted_values'])  # noqa

        data = TestHelper.get_titanic_data()

        # main reason we want to split the data is to get the means/st_devs so that we can confirm with
        # e.g. the Searcher
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.25)
        training_indexes, _ = splitter.split(target_values=data.Survived)

        train_data_y = data.iloc[training_indexes].Survived
        train_data = data.iloc[training_indexes].drop(columns='Survived')

        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]

        decorator = TempDecorator()
        resampler = RepeatedCrossValidationResampler(
            model=MockClassificationModelWrapper(data_y=data.Survived),
            transformations=None,
            scores=score_list,
            folds=5,
            repeats=2,
            fold_decorators=[decorator])
        resampler.resample(data_x=train_data, data_y=train_data_y)

        assert decorator._repeat_index == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert decorator._fold_index == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        # The _holdout_indexes should have twice the number of indexes as training_indexes because of
        # `repeats=2`
        num_holdout_indexes = len(decorator._holdout_indexes)
        num_training_indexes = len(training_indexes)
        assert num_training_indexes * 2 == num_holdout_indexes
        assert len(set(training_indexes)) == num_training_indexes

        # get the holdout indexes from the first repeat. This should contain exactly 1 to 1 indexes with the
        # original training indexes, although not in the same order
        repeat_0_holdout_indexes = decorator._holdout_indexes[0:int(num_holdout_indexes / 2)]
        assert len(repeat_0_holdout_indexes) == num_training_indexes
        # check that the training indexes and holdout indexes from the first repeat contain the same values
        assert set(training_indexes) == set(repeat_0_holdout_indexes)

        repeat_1_holdout_indexes = decorator._holdout_indexes[int(num_holdout_indexes / 2): num_holdout_indexes]  # noqa
        assert len(repeat_1_holdout_indexes) == num_training_indexes
        # check that the training indexes and holdout indexes from the second repeat contain the same values
        assert set(training_indexes) == set(repeat_1_holdout_indexes)

        # at this point we know that both repeats contain the indexes from the original training set
        # this should correspond to the indexes of the predicted values DataFrame
        # first, lets merge the indexes from repeats, and assign into a different list, also used below
        repeat_0_holdout_indexes.extend(repeat_1_holdout_indexes)
        holdout_indexes = repeat_0_holdout_indexes
        assert len(decorator._holdout_predicted_values.index.values) == len(holdout_indexes)
        assert all(decorator._holdout_predicted_values.index.values == holdout_indexes)

        # lets repeat the same procedure to verify that the indexes are the same across resampler objects
        decorator = TempDecorator()
        resampler = RepeatedCrossValidationResampler(
            model=MockClassificationModelWrapper(data_y=data.Survived),
            transformations=None,
            scores=score_list,
            folds=5,
            repeats=2,
            fold_decorators=[decorator])
        resampler.resample(data_x=train_data, data_y=train_data_y)
        # test that NEW decorator object's predicted value dataframe has the same indexes it previously did
        assert all(decorator._holdout_predicted_values.index.values == holdout_indexes)

    def test_resamplers_RandomForest_classification(self):
        data = TestHelper.get_titanic_data()

        # main reason we want to split the data is to get the means/st_devs so that we can confirm with
        # e.g. the Searcher
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.25)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.Survived
        train_data = train_data.drop(columns='Survived')

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
                          SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa

        cache_directory = TestHelper.ensure_test_directory('data/test_Resamplers/cached_test_models/test_resamplers_RandomForest_classification')  # noqa
        resampler = RepeatedCrossValidationResampler(
            model=RandomForestClassifier(),
            transformations=transformations,
            scores=score_list,
            persistence_manager=LocalCacheManager(cache_directory=cache_directory),
            folds=5,
            repeats=5)

        self.assertRaises(ModelNotFittedError, lambda: resampler.results)

        resampler.resample(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP())
        assert len(resampler.results._scores) == 25
        assert all([len(x) == 4 and
                    isinstance(x[0], KappaScore) and
                    isinstance(x[1], SensitivityScore) and
                    isinstance(x[2], SpecificityScore) and
                    isinstance(x[3], ErrorRateScore)
                    for x in resampler.results._scores])
        assert resampler.results.num_resamples == 25

        # noinspection SpellCheckingInspection
        expected_file = 'repeat{0}_fold{1}_RandomForestClassifier_n_estimators500_criteriongini_max_featuresNone_max_depthNone_min_samples_split2_min_samples_leaf1_min_weight_fraction_leaf0.0_max_leaf_nodesNone_min_impurity_decrease0_bootstrapTrue_oob_scoreFalse.pkl'  # noqa
        for fold_index in range(5):
            for repeat_index in range(5):
                assert os.path.isfile(os.path.join(cache_directory,
                                                   expected_file.format(fold_index, repeat_index)))

        assert resampler.results.metrics == ['kappa', 'sensitivity', 'specificity', 'error_rate']

        # make sure the order of the resampled_scores is the same order as Evaluators passed in
        assert all(resampler.results.resampled_scores.columns.values == ['kappa', 'sensitivity', 'specificity', 'error_rate'])  # noqa

        # metric_means and metric_standard_deviations comes from resampled_scores, so testing both
        assert isclose(resampler.results.metric_means['kappa'], 0.586495320545703)
        assert isclose(resampler.results.metric_means['sensitivity'], 0.721899136052689)
        assert isclose(resampler.results.metric_means['specificity'], 0.8617441563168404)
        assert isclose(resampler.results.metric_means['error_rate'], 0.192053148900336)

        assert isclose(resampler.results.metric_standard_deviations['kappa'], 0.06833478821655113)
        assert isclose(resampler.results.metric_standard_deviations['sensitivity'], 0.06706830388930413)
        assert isclose(resampler.results.metric_standard_deviations['specificity'], 0.03664756028501139)
        assert isclose(resampler.results.metric_standard_deviations['error_rate'], 0.031189357324296424)

        assert isclose(resampler.results.metric_coefficient_of_variation['kappa'], round(0.06833478821655113 / 0.586495320545703, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['sensitivity'], round(0.06706830388930413 / 0.721899136052689, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['specificity'], round(0.03664756028501139 / 0.8617441563168404, 2))  # noqa
        assert isclose(resampler.results.metric_coefficient_of_variation['error_rate'], round(0.031189357324296424 / 0.192053148900336, 2))  # noqa

        plt.gcf().clear()
        TestHelper.check_plot('data/test_Resamplers/test_resamplers_RandomForest_classification_cv_boxplot.png',  # noqa
                              lambda: resampler.results.resampled_scores_boxplot())

    # noinspection PyTypeChecker
    def test_resampling_roc_pr_thresholds(self):
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

        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]
        resampler = RepeatedCrossValidationResampler(
            model=RandomForestClassifier(),
            transformations=transformations,
            scores=score_list,
            folds=5,
            repeats=1,
            fold_decorators=[decorator])
        resampler.resample(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP())

        expected_roc_thresholds = [0.43, 0.31, 0.47, 0.59, 0.48]
        expected_precision_recall_thresholds = [0.43, 0.53, 0.64, 0.59, 0.6]
        assert decorator.roc_ideal_thresholds == expected_roc_thresholds
        assert decorator.precision_recall_ideal_thresholds == expected_precision_recall_thresholds
        assert isclose(decorator.roc_ideal_thresholds_mean, np.mean(expected_roc_thresholds))
        assert isclose(decorator.resampled_precision_recall_mean, np.mean(expected_precision_recall_thresholds))  # noqa
        assert isclose(decorator.roc_ideal_thresholds_st_dev, np.std(expected_roc_thresholds))
        assert isclose(decorator.resampled_precision_recall_st_dev, np.std(expected_precision_recall_thresholds))  # noqa
        assert isclose(decorator.roc_ideal_thresholds_cv, round(np.std(expected_roc_thresholds) / np.mean(expected_roc_thresholds), 2))  # noqa
        assert isclose(decorator.resampled_precision_recall_cv, round(np.std(expected_precision_recall_thresholds) / np.mean(expected_precision_recall_thresholds), 2))  # noqa

        # the object should be stored in the results as the first and only decorator element
        assert len(resampler.results.decorators) == 1
        assert resampler.results.decorators[0] is decorator  # should be the same objects

        # Test AucX (just test 2 folds, to make sure it finds `positive_class` (takes too long to test more)
        decorator = TwoClassThresholdDecorator()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [AucRocScore(positive_class=1)]
        resampler = RepeatedCrossValidationResampler(
            model=RandomForestClassifier(),
            transformations=transformations,
            scores=score_list,
            folds=2,
            repeats=1,
            fold_decorators=[decorator])
        resampler.resample(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP())
        expected_roc_thresholds = [0.35, 0.48]
        expected_precision_recall_thresholds = [0.35, 0.48]
        assert decorator.roc_ideal_thresholds == expected_roc_thresholds
        assert decorator.precision_recall_ideal_thresholds == expected_precision_recall_thresholds

        # the object should be stored in the results as the first and only decorator element
        assert len(resampler.results.decorators) == 1
        assert resampler.results.decorators[0] is decorator  # should be the same objects

        # Test DummyClassifier; utilize edge cases
        decorator = TwoClassThresholdDecorator()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [AucRocScore(positive_class=1)]
        resampler = RepeatedCrossValidationResampler(
            model=DummyClassifier(strategy=DummyClassifierStrategy.MOST_FREQUENT),
            transformations=transformations,
            scores=score_list,
            folds=2,
            repeats=1,
            fold_decorators=[decorator])
        resampler.resample(data_x=train_data, data_y=train_data_y)
        expected_roc_thresholds = [0.0, 0.0]
        expected_precision_recall_thresholds = [0.0, 0.0]
        assert decorator.roc_ideal_thresholds == expected_roc_thresholds
        assert decorator.precision_recall_ideal_thresholds == expected_precision_recall_thresholds

        # the object should be stored in the results as the first and only decorator element
        assert len(resampler.results.decorators) == 1
        assert resampler.results.decorators[0] is decorator  # should be the same objects
