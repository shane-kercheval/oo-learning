import os
import os.path
import pickle
import shutil
import warnings
from math import isclose
from os import remove
from typing import Callable
from mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning import *
from tests.MockClassificationModelWrapper import MockClassificationModelWrapper
from tests.MockRegressionModelWrapper import MockRegressionModelWrapper
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


class MockHyperParams(HyperParamsBase):

    def __init__(self):
        super().__init__()

        self._params_dict = dict(a='a', b='b', c='c')

    @property
    def test(self):
        return 'test hyper-params'


class MockDevice:
    """
    A mock device to temporarily suppress output to stdout
    Similar to UNIX /dev/null.
    http://keenhenry.me/suppress-stdout-in-unittest/
    """
    def write(self, s): pass


class MockPersistenceManagerBase(PersistenceManagerBase):
    def set_key_prefix(self, prefix: str):
        pass

    def set_key(self, key: str):
        pass

    def get_object(self, fetch_function: Callable[[], object], key: str = None):
        pass


# noinspection SpellCheckingInspection,PyUnresolvedReferences,PyMethodMayBeStatic, PyTypeChecker
class ModelWrapperTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_MockModelWrapper(self):
        ######################################################################################################
        # seems stupid to teset a Mock object, but I just want to ensure it does what I think it will do.
        ######################################################################################################

        ######################################################################################################
        # MockClassificationModelWrapper, integers
        # for Classification problems, predict should return a pd.DataFrame of probabilities, which will be
        # 0/1's for the Mock object
        ######################################################################################################
        data = TestHelper.get_titanic_data()
        np.random.seed(123)
        mock_y = np.random.choice(a=np.arange(0, 3), p=[0.1, 0.3, 0.6], size=1000)  # random target values
        mock_model = MockClassificationModelWrapper(data_y=mock_y)
        mock_model.train(data_x=data, data_y=data.Survived)
        assert mock_model._unique_targets == [2, 1, 0]
        assert mock_model._target_probabilities == [0.596, 0.306, 0.098]

        predictions = mock_model.predict(data_x=data)
        assert predictions.shape == (len(data), 3)
        # ensure similar distribution
        assert all(predictions.columns.values == [2, 1, 0])
        # means should equal the expected value percentages i.e. distribution
        assert isclose(predictions[2].mean(), 0.59259259259259256)
        assert isclose(predictions[1].mean(), 0.29741863075196406)
        assert isclose(predictions[0].mean(), 0.10998877665544332)

        predictions = mock_model.predict(data_x=data.iloc[0:100, ])
        assert predictions.shape == (100, 3)
        # ensure similar distribution
        assert all(predictions.columns.values == [2, 1, 0])
        # means should equal the expected value percentages i.e. distribution
        assert isclose(predictions[2].mean(), 0.62)
        assert isclose(predictions[1].mean(), 0.33)
        assert isclose(predictions[0].mean(), 0.05)

        ######################################################################################################
        # MockClassificationModelWrapper, strings (expecting same values as integers)
        ######################################################################################################
        np.random.seed(123)
        mock_y = np.random.choice(a=np.arange(0, 3), p=[0.1, 0.3, 0.6], size=1000)  # random target values
        lookup_y = ['a', 'b', 'c']
        mock_y = [lookup_y[x] for x in mock_y]
        mock_model = MockClassificationModelWrapper(data_y=mock_y)
        mock_model.train(data_x=data, data_y=data.Survived)
        assert mock_model._unique_targets == ['c', 'b', 'a']
        assert mock_model._target_probabilities == [0.596, 0.306, 0.098]

        predictions = mock_model.predict(data_x=data)
        assert predictions.shape == (len(data), 3)
        # ensure similar distribution
        assert all(predictions.columns.values == ['c', 'b', 'a'])
        # means should equal the expected value percentages i.e. distribution
        assert isclose(predictions['c'].mean(), 0.59259259259259256)
        assert isclose(predictions['b'].mean(), 0.29741863075196406)
        assert isclose(predictions['a'].mean(), 0.10998877665544332)

        predictions = mock_model.predict(data_x=data.iloc[0:100, ])
        assert predictions.shape == (100, 3)
        # ensure similar distribution
        # ensure similar distribution
        assert all(predictions.columns.values == ['c', 'b', 'a'])
        # means should equal the expected value percentages i.e. distribution
        assert isclose(predictions['c'].mean(), 0.62)
        assert isclose(predictions['b'].mean(), 0.33)
        assert isclose(predictions['a'].mean(), 0.05)

        ######################################################################################################
        # MockRegressionModelWrapper
        ######################################################################################################
        data = TestHelper.get_cement_data()
        mock_model = MockRegressionModelWrapper(data_y=data.strength)
        mock_model.train(data_x=data, data_y=data.strength)
        assert [(x.left, x.right) for x in mock_model._target_intervals] == [(34.438000000000002, 42.465000000000003), (26.411000000000001, 34.438000000000002), (18.384, 26.411000000000001), (10.356999999999999, 18.384), (42.465000000000003, 50.491999999999997), (50.491999999999997, 58.518999999999998), (58.518999999999998, 66.546000000000006), (2.2490000000000001, 10.356999999999999), (66.546000000000006, 74.572999999999993), (74.572999999999993, 82.599999999999994)]  # noqa
        assert mock_model._target_probabilities == [0.19029126213592232, 0.17572815533980582, 0.15145631067961166, 0.129126213592233, 0.1087378640776699, 0.0970873786407767, 0.05048543689320388, 0.043689320388349516, 0.03495145631067961, 0.018446601941747572]  # noqa

        predictions = mock_model.predict(data_x=data)
        value_distribution = pd.Series(predictions).value_counts(normalize=True)
        assert all(value_distribution.index.values == [32.0, 40.0, 24.0, 15.9, 48.1, 56.1, 64.1, 7.9, 72.1, 80.2])  # noqa
        assert all([(x, y) for x, y in zip(value_distribution.values, [0.19902913, 0.16796117, 0.1592233, 0.12718447, 0.1184466, 0.07961165, 0.04757282, 0.04466019, 0.03398058, 0.0223301])])  # noqa

    def test_ModelWrapperBase(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        train_x, train_y, test_x, test_y = TestHelper.split_train_test_regression(data, target_variable)
        ######################################################################################################
        # test predicting without training, training an already trained model, fitted_info without training
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)

        # should raise an error if only one tuning_parameters is passed in, since expecting 2 params
        self.assertRaises(ModelNotFittedError,
                          lambda: model_wrapper.predict(data_x=train_x))

        self.assertRaises(ModelNotFittedError, lambda: model_wrapper.fitted_info)

        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        assert model_wrapper.fitted_info.results_summary == 'test_summary'
        assert model_wrapper.fitted_info.hyper_params.test == 'test hyper-params'

        self.assertRaises(ModelAlreadyFittedError,
                          lambda: model_wrapper.train(data_x=train_x,
                                                      data_y=train_y,
                                                      hyper_params=MockHyperParams()))

        predictions = model_wrapper.predict(data_x=train_x)
        assert predictions is not None

        # pass in data that has different columns
        test_x.columns = ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg',
                          'column_does_not_exist']
        self.assertRaises(AssertionError,
                          model_wrapper.predict,
                          data_x=test_x)

    def test_ModelWrapperBase_caching_model(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        train_x, train_y, test_x, test_y = TestHelper.split_train_test_regression(data, target_variable)

        ######################################################################################################
        # calling `set_persistence_manager()` after `train()` should fail
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params={'test': 'test1'})
        self.assertRaises(ModelAlreadyFittedError, lambda: model_wrapper.set_persistence_manager(persistence_manager=MockPersistenceManagerBase()))  # noqa

        ######################################################################################################
        # calling `clone()` after `set_persistence_manager` should fail
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)
        model_wrapper.set_persistence_manager(persistence_manager=MockPersistenceManagerBase())
        self.assertRaises(ModelCachedAlreadyConfigured, lambda: model_wrapper.clone())

        ######################################################################################################
        # caching and it does not exist
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)
        cache_directory = TestHelper.ensure_test_directory('data/temp_caching_tests')
        cache_key = 'test_caching_file'
        file_path = os.path.join(cache_directory, cache_key + '.pkl')
        assert os.path.isdir(cache_directory) is False
        assert os.path.isfile(file_path) is False

        model_wrapper.set_persistence_manager(persistence_manager=LocalCacheManager(cache_directory=cache_directory, key=cache_key))  # noqa
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        # ensure the model "trained"
        assert model_wrapper.fitted_info.results_summary == 'test_summary'
        assert model_wrapper.fitted_info.hyper_params.test == 'test hyper-params'
        # ensure the model is now cached
        assert os.path.isfile(file_path) is True
        with open(file_path, 'rb') as saved_object:
            model_object = pickle.load(saved_object)
            assert model_object == 'test model_object'  # this is from the MockRegressionModelWrapper

        ######################################################################################################
        # caching and it already exists
        # setting `model_object` on a cached/existing model, should not be updated in the model or the cache
        ######################################################################################################
        # first ensure that setting `model_object` results in fitted_info.model_object being changed
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength, model_object='new model object!!')
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        assert model_wrapper.fitted_info.model_object == 'new model object!!'

        # now, if we pass in the same `model_object` to a previously cached model, we should get the old value
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength, model_object='new model object!!')
        assert os.path.isfile(file_path) is True  # should already exist from above
        model_wrapper.set_persistence_manager(persistence_manager=LocalCacheManager(cache_directory=cache_directory, key=cache_key))  # noqa
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        # ensure the cached value in fitted_info is the same (and not changed to 'new model object!!')
        assert model_wrapper.fitted_info.model_object == 'test model_object'  # CACHED value !!!!!
        # ensure the model "trained"
        assert model_wrapper.fitted_info.results_summary == 'test_summary'
        assert model_wrapper.fitted_info.hyper_params.test == 'test hyper-params'
        assert os.path.isfile(file_path) is True
        # ensure same cache (i.e. has old/cached model_object value)
        with open(file_path, 'rb') as saved_object:
            model_object = pickle.load(saved_object)
            assert model_object == 'test model_object'  # old model_object value ensures same cache

        os.remove(file_path)  # clean up

        ######################################################################################################
        # predicting with a cached model that does not exist (need to call `train()` before `predict()`)
        # `predict()` should not change, basically testing that we have a model via fitted_info
        # we already tested above that the correct model_object is being cached/retrieved
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)
        assert os.path.isfile(file_path) is False
        model_wrapper.set_persistence_manager(persistence_manager=LocalCacheManager(cache_directory=cache_directory, key=cache_key))  # noqa
        # fails because we have not trained
        self.assertRaises(ModelNotFittedError, lambda: model_wrapper.predict(data_x=test_x))

        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        predictions = model_wrapper.predict(data_x=test_x)  # basically just testing that it works
        # ensure predictions are close to expected via RMSE
        assert isclose(RmseEvaluator().evaluate(actual_values=test_y, predicted_values=predictions), 23.528246193289437)  # noqa

        ######################################################################################################
        # predicting with a cached model that already exists (still need to call `train()` before `predict()`,
        # because train has parameters that are needed to pass to the FittedInfo object
        # `predict()` should not change, basically testing that we have a model via fitted_info
        # we already tested above that the correct model_object is being cached/retrieved
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)
        assert os.path.isfile(file_path) is True  # left over from last section
        model_wrapper.set_persistence_manager(persistence_manager=LocalCacheManager(cache_directory=cache_directory, key=cache_key))  # noqa
        # fails because we have not trained
        self.assertRaises(ModelNotFittedError, lambda: model_wrapper.predict(data_x=test_x))

        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        # ensure predictions are close to expected via RMSE
        assert isclose(RmseEvaluator().evaluate(actual_values=test_y, predicted_values=predictions), 23.528246193289437)  # noqa

        os.remove(file_path)  # clean up

        shutil.rmtree(cache_directory)

    def test_HyperParamsBase(self):
        params = MockHyperParams()
        assert params.params_dict == dict(a='a', b='b', c='c')
        params.update_dict(dict(b=1, c=None))
        assert params.params_dict == dict(a='a', b=1, c=None)
        params.update_dict(dict(b=1, c=None))

        # cannot update non-existant hyper-param (d)
        self.assertRaises(ValueError, lambda: params.update_dict(dict(d='d')))

    def test_RegressionMW(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        data_x = data.drop(target_variable, axis=1)
        data_y = data.strength

        model = RegressionMW()
        model.train(data_x=data_x, data_y=data_y)
        assert model.fitted_info.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic',
                                                   'coarseagg', 'fineagg', 'age']

        # TODO: test warnings
        # TODO: test feature_importance

        assert isclose(10.399142639503246, model.fitted_info.summary_stats['residual standard error (RSE)'])
        assert isclose(0.61250729349293431, model.fitted_info.summary_stats['adjusted r-squared'])
        assert isclose(6.2859778203065082e-206, model.fitted_info.summary_stats['model p-value'])
        assert isclose(0.62279152099417079, model.fitted_info.summary_stats['Ratio RSE to Target STD'])
        assert isclose(0.064673395463680838, model.fitted_info.summary_stats['Residual Correlations'])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_concrete_regression_results.pkl'))  # noqa
        # SAVE THE ORIGINAL RESULTS, WHICH WERE VERIFIED AGAINST R'S LM() FUNCTION
        # with open(file, 'wb') as output:
        #     pickle.dump(model.results_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            regression_results_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=regression_results_summary,
                                                      data_frame2=model.fitted_info.results_summary)

        predictions = model.predict(data_x=data_x)

        assert isclose(RmseEvaluator().evaluate(actual_values=data_y, predicted_values=predictions),
                       10.353609808895648)

        # test ROC curve
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMW_regression_plots.png'))  # noqa
        assert os.path.isfile(file)
        remove(file)
        assert os.path.isfile(file) is False
        # noinspection PyStatementEffect
        model.fitted_info.graph
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

    def test_ModelFitter_transformations(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        ######################################################################################################
        # create missing values in 'random' rows of cement/ash; ensure it is consistent across runs
        # create a categorical feature to test dummy encoding
        ######################################################################################################
        np.random.seed(42)
        missing_indexes_cement = np.random.randint(low=0, high=len(data), size=int(len(data) * 0.10))
        data.loc[missing_indexes_cement, 'cement'] = None

        np.random.seed(43)
        missing_indexes_ash = np.random.randint(low=0, high=len(data), size=int(len(data) * 0.10))
        data.loc[missing_indexes_ash, 'ash'] = None

        np.random.seed(42)
        random_codes = np.random.randint(low=0, high=2, size=len(data))
        data['random'] = ['code0' if random_code == 0 else 'code1' for random_code in random_codes]

        ######################################################################################################
        #  split up data, determine indexes of training/test sets where we have missing values for cement/ash
        ######################################################################################################
        train_x, train_y, test_x, test_y = TestHelper.split_train_test_regression(data, target_variable)
        expected_cement_median = train_x['cement'].median()
        expected_ash_median = train_x['ash'].median()

        index_missing_train_cement = list(set(train_x.index.values).intersection(set(missing_indexes_cement)))
        index_missing_test_cement = list(set(test_x.index.values).intersection(set(missing_indexes_cement)))

        index_missing_train_ash = list(set(train_x.index.values).intersection(set(missing_indexes_ash)))
        index_missing_test_ash = list(set(test_x.index.values).intersection(set(missing_indexes_ash)))
        ######################################################################################################
        # ensure that all the indexes that we expect are missing values
        ######################################################################################################
        assert all(train_x.loc[index_missing_train_cement]['cement'].isnull())
        assert all(train_x.loc[index_missing_train_ash]['ash'].isnull())
        assert all(test_x.loc[index_missing_test_cement]['cement'].isnull())
        assert all(test_x.loc[index_missing_test_ash]['ash'].isnull())

        ######################################################################################################
        # fit/predict the model using the Mock object, which stores the transformed training/test data
        # so we can validate the expected transformations took place across both datasets
        ######################################################################################################
        evaluators = [RmseEvaluator(), MaeEvaluator()]
        model_fitter = ModelFitter(model=MockRegressionModelWrapper(data_y=data.strength),
                                   model_transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),
                                                          ImputationTransformer(),
                                                          DummyEncodeTransformer()],
                                   evaluators=evaluators)

        # should raise an error calling `predict` before `fit`
        self.assertRaises(ModelNotFittedError, lambda: model_fitter.predict(data_x=test_x))

        model_fitter.fit(data_x=train_x, data_y=train_y, hyper_params=None)  # mock object stores transformed
        assert all([x is y for x, y in zip(model_fitter.training_evaluators, evaluators)])
        expected_accuracies = [24.50861705505752, 19.700946601941748]
        assert all([isclose(x, y) for x, y in zip(model_fitter.training_accuracies, expected_accuracies)])

        # should not be able to call fit twice
        self.assertRaises(ModelAlreadyFittedError, lambda: model_fitter.fit(data_x=train_x,
                                                                            data_y=train_y,
                                                                            hyper_params=None))

        predictions = model_fitter.predict(data_x=test_x)  # mock object stores transformed data
        assert predictions is not None
        ######################################################################################################
        # removed coarseagg and fineagg, added a categorical column and used DUMMY encoding
        ######################################################################################################
        assert model_fitter.model_info.feature_names == \
            ['cement', 'slag', 'ash', 'water', 'superplastic', 'age', 'random_code1']
        ######################################################################################################
        # ensure that we imputed the correct values in the correct indexes
        ######################################################################################################
        # training set, the Mock model wrapper saves the training data in `fitted_train_x` field, and test_x
        # so we can 'peak' inside and see the transformations

        # ensure transformation states are set correctly
        assert model_fitter._model_transformations.transformations[0].state == {}
        assert model_fitter._model_transformations.transformations[1].state == \
            {'cement': 266.19999999999999,
             'slag': 26.0,
             'ash': 0.0,
             'water': 185.69999999999999,
             'superplastic': 6.4000000000000004,
             'age': 28.0,
             'random': 'code1'}
        assert model_fitter._model_transformations.transformations[2].state == {'random': ['code0', 'code1']}

        # ensure the data is updated/imputed correctly
        assert all(model_fitter._model.fitted_train_x.loc[index_missing_train_cement]['cement'] ==
                   expected_cement_median)
        assert all(model_fitter._model.fitted_train_x.loc[index_missing_train_ash]['ash'] ==
                   expected_ash_median)
        # test set
        assert all(model_fitter._model.fitted_test_x.loc[index_missing_test_cement]['cement'] ==
                   expected_cement_median)
        assert all(model_fitter._model.fitted_test_x.loc[index_missing_test_ash]['ash'] ==
                   expected_ash_median)
        ######################################################################################################
        # ensure that we calculated the correct dummy encodings
        ######################################################################################################
        # numeric indexes associated with code1
        training_code1 = [index for index in train_x.index.values if train_x.loc[index]['random'] == 'code1']
        # boolean indexes that correspond with code1 (so we can negate)
        indexes = model_fitter._model.fitted_train_x.index.isin(training_code1)
        assert all(model_fitter._model.fitted_train_x[indexes]['random_code1'] == 1)
        assert all(model_fitter._model.fitted_train_x[~indexes]['random_code1'] == 0)

        # same for test set
        test_code1 = [index for index in test_x.index.values if test_x.loc[index]['random'] == 'code1']
        # boolean indexes that correspond with code1 (so we can negate)
        indexes = model_fitter._model.fitted_test_x.index.isin(test_code1)
        assert all(model_fitter._model.fitted_test_x[indexes]['random_code1'] == 1)
        assert all(model_fitter._model.fitted_test_x[~indexes]['random_code1'] == 0)
        ######################################################################################################
        # ensure that we didn't change any of the original datasets
        ######################################################################################################
        assert all(train_x.loc[index_missing_train_cement]['cement'].isnull())
        assert all(train_x.loc[index_missing_train_ash]['ash'].isnull())
        assert all(test_x.loc[index_missing_test_cement]['cement'].isnull())
        assert all(test_x.loc[index_missing_test_ash]['ash'].isnull())

    def test_RegressionMW_with_ModelFitter(self):
        data = TestHelper.get_cement_data()

        fitter = ModelFitter(model=RegressionMW(),
                             model_transformations=ModelDefaults.transformations_regression(),
                             evaluators=[RmseEvaluator()])

        fitter.fit(data_x=data.drop('strength', axis=1), data_y=data['strength'])

        # TODO: test warnings
        # TODO: test feature_importance

        assert fitter.model_info.hyper_params is None

        assert len(fitter.training_evaluators) == 1
        assert len(fitter.training_accuracies) == 1
        assert isclose(10.353609808895648, fitter.training_evaluators[0].value)
        assert isclose(10.353609808895648, fitter.training_accuracies[0])
        assert isclose(10.399142639503246,
                       fitter.model_info.summary_stats['residual standard error (RSE)'])
        assert isclose(0.61250729349293431, fitter.model_info.summary_stats['adjusted r-squared'])
        assert isclose(6.2859778203065082e-206, fitter.model_info.summary_stats['model p-value'])
        assert isclose(0.62279152099417079, fitter.model_info.summary_stats['Ratio RSE to Target STD'])
        assert isclose(0.064673395463680838, fitter.model_info.summary_stats['Residual Correlations'])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_concrete_regression_results.pkl'))  # noqa
        # SAVE THE ORIGINAL RESULTS, WHICH WERE VERIFIED AGAINST R'S LM() FUNCTION
        # with open(file, 'wb') as output:
        #     pickle.dump(fitter.results_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            regression_results_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=regression_results_summary,
                                                      data_frame2=fitter.model_info.results_summary)
        ######################################################################################################
        # check prediction values
        ######################################################################################################
        predictions = fitter.predict(data_x=data.drop('strength', axis=1))
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_expected_predictions.csv'))  # noqa
        # predictions.to_csv(file)
        expected_predictions = pd.read_csv(file)
        expected_predictions = expected_predictions.predictions
        assert all([isclose(x, y) for x, y in zip(predictions, expected_predictions)])
        assert RmseEvaluator().evaluate(actual_values=data['strength'], predicted_values=predictions) == \
            fitter.training_accuracies[0]
        assert isinstance(fitter.model_info.warnings, dict)
        assert len(fitter.model_info.warnings) == 0
        assert fitter.model_info.graph is not None

        ######################################################################################################
        # same thing, but with MaeEvaluator, which shouldn't change any values except for the value
        ######################################################################################################
        fitter = ModelFitter(model=RegressionMW(),
                             model_transformations=ModelDefaults.transformations_regression(),
                             evaluators=[MaeEvaluator()])

        fitter.fit(data_x=data.drop('strength', axis=1), data_y=data['strength'])

        # TODO: test warnings
        # TODO: test feature_importance

        assert fitter.model_info.hyper_params is None
        assert isclose(8.2143437062218183, fitter.training_evaluators[0].value)
        assert isclose(8.2143437062218183, fitter.training_accuracies[0])
        assert isclose(10.399142639503246,
                       fitter.model_info.summary_stats['residual standard error (RSE)'])
        assert isclose(0.61250729349293431, fitter.model_info.summary_stats['adjusted r-squared'])
        assert isclose(6.2859778203065082e-206, fitter.model_info.summary_stats['model p-value'])
        assert isclose(0.62279152099417079, fitter.model_info.summary_stats['Ratio RSE to Target STD'])
        assert isclose(0.064673395463680838, fitter.model_info.summary_stats['Residual Correlations'])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_concrete_regression_results.pkl'))  # noqa
        # SAVE THE ORIGINAL RESULTS, WHICH WERE VERIFIED AGAINST R'S LM() FUNCTION
        # with open(file, 'wb') as output:
        #     pickle.dump(fitter.results_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            regression_results_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=regression_results_summary,
                                                      data_frame2=fitter.model_info.results_summary)
        ######################################################################################################
        # check prediction values
        ######################################################################################################
        predictions = fitter.predict(data_x=data.drop('strength', axis=1))
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_expected_predictions.csv'))  # noqa
        # predictions.to_csv(file)
        expected_predictions = pd.read_csv(file)
        expected_predictions = expected_predictions.predictions
        assert all([isclose(x, y) for x, y in zip(predictions, expected_predictions)])
        assert MaeEvaluator().evaluate(actual_values=data['strength'], predicted_values=predictions) == \
            fitter.training_accuracies
        assert isinstance(fitter.model_info.warnings, dict)
        assert len(fitter.model_info.warnings) == 0
        assert fitter.model_info.graph is not None

        # Evaluate with same data, make sure holdout data matches training data

        ######################################################################################################
        # test transformations
        # RegressionMW does Imputation and Dummy Encoding transformations
        ######################################################################################################
        trans_data = data.copy()
        target_variable = 'strength'

        np.random.seed(42)
        missing_indexes_cement = np.random.randint(low=0, high=len(data), size=int(len(data)*0.10))
        trans_data.loc[missing_indexes_cement, 'cement'] = None

        np.random.seed(43)
        missing_indexes_ash = np.random.randint(low=0, high=len(data), size=int(len(data)*0.10))
        trans_data.loc[missing_indexes_ash, 'ash'] = None

        np.random.seed(42)
        random_codes = np.random.randint(low=0, high=2, size=len(data))
        trans_data['random'] = ['code0' if random_code == 0 else 'code1' for random_code in random_codes]

        train_x, train_y, test_x, test_y = TestHelper.split_train_test_regression(trans_data, target_variable)
        expected_cement_median = 266.2  # trans_data['cement'].median()
        expected_ash_median = 0  # trans_data['ash'].median()

        ######################################################################################################
        # passing in missing data should raise MissingValueError
        ######################################################################################################
        fitter = ModelFitter(model=RegressionMW(),
                             model_transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg'])],
                             evaluators=[RmseEvaluator()])
        # should raise an error because our transformations didn't impute missing values
        self.assertRaises(MissingValueError,
                          fitter.fit,
                          data_x=trans_data.drop('strength', axis=1), data_y=trans_data['strength'])

        ######################################################################################################
        # check transformations
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['coarseagg', 'fineagg'])] + \
            ModelDefaults.transformations_regression()
        fitter = ModelFitter(model=RegressionMW(),
                             model_transformations=transformations,
                             evaluators=[RmseEvaluator()])
        fitter.fit(data_x=train_x, data_y=train_y)

        # removed coarseagg and fineagg, added a categorical column and used DUMMY encoding
        assert fitter.model_info.feature_names == \
            ['cement', 'slag', 'ash', 'water', 'superplastic', 'age', 'random_code1']

        # make sure the transformations used the correct/expected median values for imputation
        assert isclose(fitter._model_transformations.transformations[1].state['cement'],
                       expected_cement_median)
        assert fitter._model_transformations.transformations[1].state['ash'] == expected_ash_median
        assert len(fitter.training_evaluators) == 1
        assert len(fitter.training_accuracies) == 1
        assert isclose(10.965933264096071, fitter.training_accuracies[0])
        assert isclose(11.019556729421033, fitter.model_info.summary_stats['residual standard error (RSE)'])
        assert isclose(0.56860223899440199, fitter.model_info.summary_stats['adjusted r-squared'])
        assert isclose(8.3625713634704176e-146, fitter.model_info.summary_stats['model p-value'])

        ######################################################################################################
        # ensure that we didn't change any of the original datasets
        # SAVE THE ORIGINAL RESULTS, WHICH WERE VERIFIED AGAINST R'S LM() FUNCTION
        # I DID THE TRANSFORMATIONS MANUALLY, THEN EXPORTED TO A CSV, THEN LOADED INTO R, USED LM(), AND DIFF
        ######################################################################################################
        file = os.path.join(os.getcwd(),
                            TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_concrete_regression_with_transform_results.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(regression_model.results_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            regression_results_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=regression_results_summary,
                                                      data_frame2=fitter.model_info.results_summary)

        # I tested this in R's lm() function by doing manual transformations on test set then saving to csv
        expected_test_rmse = 10.609518962806746
        expected_test_mae = 8.4974379338447736

        predictions = fitter.predict(data_x=test_x).as_matrix()
        file = os.path.join(os.getcwd(),
                            TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_concrete_regression_predictions.csv'))  # noqa
        expected_predictions = np.genfromtxt(file, delimiter=',')

        assert all([isclose(x, y) for x, y in zip(expected_predictions, predictions)])

        assert isclose(RmseEvaluator().evaluate(actual_values=test_y, predicted_values=predictions),
                       expected_test_rmse)

        assert isclose(MaeEvaluator().evaluate(actual_values=test_y, predicted_values=predictions),
                       expected_test_mae)

    def test_LogisticMW(self):
        warnings.filterwarnings("ignore")
        # noinspection PyUnusedLocal
        with patch('sys.stdout', new=MockDevice()) as fake_out:  # supress output of logistic model
            data = TestHelper.get_titanic_data()
            splitter = ClassificationStratifiedDataSplitter(test_ratio=0.20)
            training_indexes, test_indexes = splitter.split(target_values=data.Survived)

            train_data = data.iloc[training_indexes]
            train_data_y = train_data.Survived
            train_data = train_data.drop('Survived', axis=1)

            test_data = data.iloc[test_indexes]
            test_data_y = test_data.Survived
            test_data = test_data.drop('Survived', axis=1)

            transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                               CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch'])] + \
                ModelDefaults.transformations_logistic()

            # test with custom threshold of 0.5
            fitter = ModelFitter(model=LogisticMW(),
                                 model_transformations=transformations,
                                 evaluators=[KappaEvaluator(positive_category=1,
                                                            negative_category=0,
                                                            use_probabilities=True,
                                                            threshold=0.5)])
            fitter.fit(data_x=train_data, data_y=train_data_y)

            # test ROC curve for training evaluator
            file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_LogisticMW_training_ROC_custom_threshold.png'))  # noqa
            assert os.path.isfile(file)
            remove(file)
            assert os.path.isfile(file) is False
            fitter.training_evaluators[0].get_roc_curve()
            plt.savefig(file)
            plt.gcf().clear()
            assert os.path.isfile(file)

            assert fitter.training_evaluators[0].confusion_matrix.all_quality_metrics == \
                {'Kappa': 0.58877590597123142, 'Two-Class Accuracy': 0.80758426966292129,
                 'Error Rate': 0.19241573033707865, 'Sensitivity': 0.7216117216117216,
                 'Specificity': 0.86104783599088841, 'False Positive Rate': 0.13895216400911162,
                 'False Negative Rate': 0.2783882783882784, 'Positive Predictive Value': 0.76356589147286824,
                 'Negative Predictive Value': 0.83259911894273131, 'Prevalence': 0.38342696629213485,
                 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}

            # fitter.evaluate_holdout(self, holdout_x, holdout_y, evaluator=None)
            accuracy = fitter.evaluate_holdout(holdout_x=test_data, holdout_y=test_data_y)
            assert isclose(accuracy[0], 0.58794854434664856)

            # test ROC curve for holdout evaluator
            file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_LogisticMW_holdout_ROC_custom_threshold.png'))  # noqa
            assert os.path.isfile(file)
            remove(file)
            assert os.path.isfile(file) is False
            fitter.holdout_evaluators[0].get_roc_curve()
            plt.savefig(file)
            plt.gcf().clear()
            assert os.path.isfile(file)

            # Should be the same as the training value
            accuracy = fitter.evaluate_holdout(holdout_x=train_data, holdout_y=train_data_y,
                                               evaluators=[KappaEvaluator(positive_category=1,
                                                                          negative_category=0,
                                                                          use_probabilities=True,
                                                                          threshold=0.5)])
            assert isclose(accuracy[0], 0.58877590597123142)

            # now assert test set
            accuracy = fitter.evaluate_holdout(holdout_x=test_data,
                                               holdout_y=test_data_y,
                                               evaluators=[AucEvaluator(positive_category=1,
                                                                        negative_category=0,
                                                                        use_probabilities=True,
                                                                        threshold=0.5)])
            assert isclose(accuracy[0], 0.85151515151515156)

            ##################################################################################################
            # check with threshold set to None, so that evaluator will find the "best" threshold.
            ##################################################################################################
            transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                               CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch'])] + \
                ModelDefaults.transformations_logistic()

            # from oolearning.transformers.TransformerPipeline import TransformerPipeline
            # pipeline = TransformerPipeline(transformations)
            # transformed = pipeline.fit_transform(data=data)
            # transformed.to_csv('~/Desktop/temptemp.csv')

            fitter = ModelFitter(model=LogisticMW(),
                                 model_transformations=transformations,
                                 evaluators=[KappaEvaluator(positive_category=1,
                                                            negative_category=0,
                                                            use_probabilities=True,
                                                            # threshold=0.5
                                                            threshold=None)])

            fitter.fit(data_x=train_data, data_y=train_data_y)

            file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_LogisticMW_training_ROC_ideal_threshold.png'))  # noqa
            assert os.path.isfile(file)
            remove(file)
            assert os.path.isfile(file) is False
            fitter.training_evaluators[0].get_roc_curve()
            plt.savefig(file)
            plt.gcf().clear()
            assert os.path.isfile(file)

            assert isclose(fitter.training_accuracies[0], 0.59052420341637712)
            assert fitter.training_evaluators[0].confusion_matrix.all_quality_metrics == \
                {'Kappa': 0.59052420341637712, 'Two-Class Accuracy': 0.8019662921348315,
                 'Error Rate': 0.19803370786516855, 'Sensitivity': 0.80219780219780223,
                 'Specificity': 0.80182232346241455, 'False Positive Rate': 0.19817767653758542,
                 'False Negative Rate': 0.19780219780219779, 'Positive Predictive Value': 0.71568627450980393,
                 'Negative Predictive Value': 0.86699507389162567, 'Prevalence': 0.38342696629213485,
                 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}

    def test_LogisticMW_string_target(self):
        data = TestHelper.get_titanic_data()
        positive_class = 'lived'
        negative_class = 'died'
        # Test with a string target variable rather than 0/1
        data.Survived = np.where(data.Survived == 1, positive_class, negative_class)

        splitter = ClassificationStratifiedDataSplitter(test_ratio=0.20)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.Survived
        train_data = train_data.drop('Survived', axis=1)

        test_data = data.iloc[test_indexes]
        test_data_y = test_data.Survived
        test_data = test_data.drop('Survived', axis=1)

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch'])] + \
            ModelDefaults.transformations_logistic()

        # test with custom threshold of 0.5
        fitter = ModelFitter(model=LogisticMW(),
                             model_transformations=transformations,
                             evaluators=[KappaEvaluator(positive_category=positive_class,
                                                        negative_category=negative_class,
                                                        use_probabilities=True,
                                                        threshold=0.5)])
        fitter.fit(data_x=train_data, data_y=train_data_y)

        # should be the same value etc as the previous test (when target values were 0/1)
        assert fitter.training_evaluators[0].confusion_matrix.all_quality_metrics == \
            {'Kappa': 0.58877590597123142, 'Two-Class Accuracy': 0.80758426966292129,
             'Error Rate': 0.19241573033707865, 'Sensitivity': 0.7216117216117216,
             'Specificity': 0.86104783599088841, 'False Positive Rate': 0.13895216400911162,
             'False Negative Rate': 0.2783882783882784, 'Positive Predictive Value': 0.76356589147286824,
             'Negative Predictive Value': 0.83259911894273131, 'Prevalence': 0.38342696629213485,
             'No Information Rate': 0.6165730337078652, 'Total Observations': 712}

        # should be the same value etc as the previous test (when target values were 0/1)
        accuracy = fitter.evaluate_holdout(holdout_x=test_data, holdout_y=test_data_y)
        assert isclose(accuracy[0], 0.58794854434664856)

        predictions = fitter.predict(data_x=test_data)
        assert 'died' in predictions.columns.values
        assert 'lived' in predictions.columns.values

    def test_RandomForestHP_classification(self):
        # ensure n_features should be passed in OR all others, but not both
        self.assertRaises(AssertionError, lambda: RandomForestHP(criterion='gini',
                                                                 num_features=100,
                                                                 max_features=100))
        # ensure invalid criterion tune_results in assertion error
        self.assertRaises(ValueError, lambda: RandomForestHP(criterion='adsf'))

        assert ~RandomForestHP(criterion='gini').is_regression
        assert ~RandomForestHP(criterion='entropy').is_regression

        # ensure passing only `criterion` works
        assert RandomForestHP(criterion='gini').params_dict == {
            'n_estimators': 500,
            'criterion': 'gini',
            'max_features': None,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42}

        # ensure passing only `criterion` works
        assert RandomForestHP(criterion='entropy').params_dict == {
            'n_estimators': 500,
            'criterion': 'entropy',
            'max_features': None,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42}

        assert RandomForestHP(criterion='gini', num_features=101).params_dict == {
            'n_estimators': 500,
            'criterion': 'gini',
            'max_features': 10,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42}

    def test_RandomForestHP_regression(self):
        # ensure n_features should be passed in OR all others, but not both
        self.assertRaises(AssertionError, lambda: RandomForestHP(criterion='mse',
                                                                 num_features=100,
                                                                 max_features=100))
        # ensure invalid criterion tune_results in assertion error
        self.assertRaises(ValueError, lambda: RandomForestHP(criterion='adsf'))

        assert RandomForestHP(criterion='MSE').is_regression
        assert RandomForestHP(criterion='MAE').is_regression

        # ensure passing only `criterion` works
        assert RandomForestHP(criterion='MSE').params_dict == {
            'n_estimators': 500,
            'criterion': 'mse',
            'max_features': None,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42}

        # ensure passing only `criterion` works
        assert RandomForestHP(criterion='MAE').params_dict == {
            'n_estimators': 500,
            'criterion': 'mae',
            'max_features': None,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42}

        assert RandomForestHP(criterion='MSE', num_features=101).params_dict == {
            'n_estimators': 500,
            'criterion': 'mse',
            'max_features': 34,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42}

    def test_RandomForestMW_classification(self):
        data = TestHelper.get_titanic_data()

        splitter = ClassificationStratifiedDataSplitter(test_ratio=0.20)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.Survived
        train_data = train_data.drop('Survived', axis=1)

        test_data = data.iloc[test_indexes]
        test_data_y = test_data.Survived
        test_data = test_data.drop('Survived', axis=1)

        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        positive_class = 1
        negative_class = 0

        cache_directory = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_RandomForestMW_classification')  # noqa

        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             evaluators=[KappaEvaluator(positive_category=positive_class,
                                                        negative_category=negative_class,
                                                        threshold=0.5)],
                             persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        assert fitter._persistence_manager._cache_directory == cache_directory
        fitter.fit(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP(criterion='gini'))
        assert os.path.isfile(fitter._persistence_manager._cache_path)

        assert fitter.model_info.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3',
                                                   'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2',
                                                   'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0',
                                                   'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5',
                                                   'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
        assert fitter.model_info.hyper_params.params_dict == {'n_estimators': 500,
                                                              'criterion': 'gini',
                                                              'max_features': None,
                                                              'max_depth': None,
                                                              'min_samples_split': 2,
                                                              'min_samples_leaf': 1,
                                                              'min_weight_fraction_leaf': 0.0,
                                                              'max_leaf_nodes': None,
                                                              'min_impurity_decrease': 0,
                                                              'bootstrap': True,
                                                              'oob_score': False,
                                                              'n_jobs': -1,
                                                              'random_state': 42}
        assert fitter.training_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.training_accuracies[0], 0.96420581655480975)

        predictions = fitter.predict(data_x=test_data)
        assert 0 in predictions.columns.values
        assert 1 in predictions.columns.values
        pos_predictions = predictions.loc[:, 1]
        assert all([isclose(round(x, 8), y) for x, y in zip(pos_predictions,
                                                            [0.216, 0.05533333, 0.086, 0.1773, 0.844,
                                                             0.22, 0.942, 0.917, 0.542, 0.15913333,
                                                             0.30118571, 0.18383333, 0.696, 0.346, 0.306,
                                                             0.11148095, 0.016, 0., 0.066, 0.996,
                                                             0., 0.886, 0.105, 0.542, 0.18133333,
                                                             1., 0.30118571, 0.224, 0.15244011, 0.11913333,
                                                             0.11394444, 0.996, 0.672, 0.02533333, 0.2775,
                                                             0.669, 0.2115, 0.05, 0.30118571, 1.,
                                                             0.1, 0.538, 0.07712381, 0.998, 0.902,
                                                             0.031, 0.056, 0.108, 0.962, 0.996,
                                                             1., 0.072, 0.96313454, 0.011, 0.464,
                                                             0.052, 0., 0.92113333, 0.896, 0.85723333,
                                                             0.166, 0.962, 0.19995714, 0.09093333, 0.704,
                                                             0.276, 1., 0.28023333, 0.07, 0.,
                                                             0.938, 0.00229286, 0.80952261, 1., 0.18760159,
                                                             0.02556667, 0.086, 0.71729048, 0.02, 0.008,
                                                             0.716, 0.05, 0.20899048, 0.37249048, 0.96876984,
                                                             0.672, 0.06592381, 0.666, 0.10856667, 1.,
                                                             0.072, 0.058, 0., 0., 0.446,
                                                             0.996, 0.864, 0.30118571, 0.086, 0.006,
                                                             0.046, 0.91, 0.82076578, 0.978, 0.,
                                                             0.114, 0.28892381, 0., 0.0325, 0.688,
                                                             0.30118571, 0.03, 0.26585714, 0.262, 0.968,
                                                             0.30118571, 0.52, 0.11035714, 0.042, 0.046,
                                                             0., 0.882, 0.155, 0.046, 0.03530952,
                                                             0.33, 0.436, 0.258, 0.004, 0.896,
                                                             0.924, 0.24622381, 0.844, 0., 0.032,
                                                             0.007, 0.952, 0.30118571, 0.84994984, 0.004,
                                                             0.182, 0.21562828, 0.638, 0.788, 0.23954603,
                                                             0.114, 0., 0.124, 0.99, 0.99,
                                                             0.032, 0.71233333, 0.932, 0.398, 0.426,
                                                             0.634, 0.058, 0.1, 0.286, 0.678,
                                                             0.386, 0.942, 0.84994984, 0.95, 0.704,
                                                             0.272, 0.133, 0.14, 1., 0.294,
                                                             0.08741667, 0.00229286, 0.1719, 0.1, 0.832,
                                                             0.014, 0.012, 0.99, 0.49186667])])

        fitter.evaluate_holdout(holdout_x=test_data, holdout_y=test_data_y)
        assert fitter.holdout_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.holdout_accuracies[0], 0.581636060100167)

        ######################################################################################################
        # test custom hyper-parameters
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             evaluators=[KappaEvaluator(positive_category=1,
                                                        negative_category=0,
                                                        use_probabilities=True,
                                                        threshold=0.5)])
        fitter.fit(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP(criterion='gini',
                                                                                       max_features='auto',
                                                                                       n_estimators=10,
                                                                                       n_jobs=2))

        assert fitter.model_info.hyper_params.params_dict == {'n_estimators': 10,
                                                              'criterion': 'gini',
                                                              'max_features': 'auto',
                                                              'max_depth': None,
                                                              'min_samples_split': 2,
                                                              'min_samples_leaf': 1,
                                                              'min_weight_fraction_leaf': 0.0,
                                                              'max_leaf_nodes': None,
                                                              'min_impurity_decrease': 0,
                                                              'bootstrap': True,
                                                              'oob_score': False,
                                                              'n_jobs': 2,
                                                              'random_state': 42}

        assert fitter.training_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.training_accuracies[0], 0.94918555835432417)

        fitter.evaluate_holdout(holdout_x=test_data, holdout_y=test_data_y)
        assert fitter.holdout_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.holdout_accuracies[0], 0.55034286102247276)

        ######################################################################################################
        # test custom hyper-parameters, num_features
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             evaluators=[KappaEvaluator(positive_category=1,
                                                        negative_category=0,
                                                        use_probabilities=True,
                                                        threshold=0.5)])
        fitter.fit(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP(criterion='gini',
                                                                                       num_features=11))

        assert fitter.model_info.hyper_params.params_dict == {'n_estimators': 500,
                                                              'criterion': 'gini',
                                                              'max_features': 3,
                                                              'max_depth': None,
                                                              'min_samples_split': 2,
                                                              'min_samples_leaf': 1,
                                                              'min_weight_fraction_leaf': 0.0,
                                                              'max_leaf_nodes': None,
                                                              'min_impurity_decrease': 0,
                                                              'bootstrap': True,
                                                              'oob_score': False,
                                                              'n_jobs': -1,
                                                              'random_state': 42}

        assert fitter.training_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.training_accuracies[0], 0.96420581655480975)

        fitter.evaluate_holdout(holdout_x=test_data, holdout_y=test_data_y)
        assert fitter.holdout_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.holdout_accuracies[0], 0.61644569438864327)

    def test_RandomForestMW_classification_string_target(self):
        data = TestHelper.get_titanic_data()
        positive_class = 'lived'
        negative_class = 'died'

        # Test with a string target variable rather than 0/1
        data.Survived = np.where(data.Survived == 1, positive_class, negative_class)

        splitter = ClassificationStratifiedDataSplitter(test_ratio=0.20)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.Survived
        train_data = train_data.drop('Survived', axis=1)

        test_data = data.iloc[test_indexes]
        test_data_y = test_data.Survived
        test_data = test_data.drop('Survived', axis=1)

        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             evaluators=[KappaEvaluator(positive_category=positive_class,
                                                        negative_category=negative_class,
                                                        threshold=0.5)])
        fitter.fit(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP(criterion='gini'))
        assert fitter.model_info.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3',
                                                   'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2',
                                                   'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0',
                                                   'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5',
                                                   'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

        assert fitter.model_info.hyper_params.params_dict == {'n_estimators': 500,
                                                              'criterion': 'gini',
                                                              'max_features': None,
                                                              'max_depth': None,
                                                              'min_samples_split': 2,
                                                              'min_samples_leaf': 1,
                                                              'min_weight_fraction_leaf': 0.0,
                                                              'max_leaf_nodes': None,
                                                              'min_impurity_decrease': 0,
                                                              'bootstrap': True,
                                                              'oob_score': False,
                                                              'n_jobs': -1,
                                                              'random_state': 42}
        assert fitter.training_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.training_accuracies[0], 0.96420581655480975)

        predictions = fitter.predict(data_x=test_data)
        assert positive_class in predictions.columns.values
        assert negative_class in predictions.columns.values
        pos_predictions = predictions.loc[:, positive_class]
        assert all([isclose(round(x, 8), y) for x, y in zip(pos_predictions,
                                                            [0.216, 0.05533333, 0.086, 0.1773, 0.844,
                                                             0.22, 0.942, 0.917, 0.542, 0.15913333,
                                                             0.30118571, 0.18383333, 0.696, 0.346, 0.306,
                                                             0.11148095, 0.016, 0., 0.066, 0.996,
                                                             0., 0.886, 0.105, 0.542, 0.18133333,
                                                             1., 0.30118571, 0.224, 0.15244011, 0.11913333,
                                                             0.11394444, 0.996, 0.672, 0.02533333, 0.2775,
                                                             0.669, 0.2115, 0.05, 0.30118571, 1.,
                                                             0.1, 0.538, 0.07712381, 0.998, 0.902,
                                                             0.031, 0.056, 0.108, 0.962, 0.996,
                                                             1., 0.072, 0.96313454, 0.011, 0.464,
                                                             0.052, 0., 0.92113333, 0.896, 0.85723333,
                                                             0.166, 0.962, 0.19995714, 0.09093333, 0.704,
                                                             0.276, 1., 0.28023333, 0.07, 0.,
                                                             0.938, 0.00229286, 0.80952261, 1., 0.18760159,
                                                             0.02556667, 0.086, 0.71729048, 0.02, 0.008,
                                                             0.716, 0.05, 0.20899048, 0.37249048, 0.96876984,
                                                             0.672, 0.06592381, 0.666, 0.10856667, 1.,
                                                             0.072, 0.058, 0., 0., 0.446,
                                                             0.996, 0.864, 0.30118571, 0.086, 0.006,
                                                             0.046, 0.91, 0.82076578, 0.978, 0.,
                                                             0.114, 0.28892381, 0., 0.0325, 0.688,
                                                             0.30118571, 0.03, 0.26585714, 0.262, 0.968,
                                                             0.30118571, 0.52, 0.11035714, 0.042, 0.046,
                                                             0., 0.882, 0.155, 0.046, 0.03530952,
                                                             0.33, 0.436, 0.258, 0.004, 0.896,
                                                             0.924, 0.24622381, 0.844, 0., 0.032,
                                                             0.007, 0.952, 0.30118571, 0.84994984, 0.004,
                                                             0.182, 0.21562828, 0.638, 0.788, 0.23954603,
                                                             0.114, 0., 0.124, 0.99, 0.99,
                                                             0.032, 0.71233333, 0.932, 0.398, 0.426,
                                                             0.634, 0.058, 0.1, 0.286, 0.678,
                                                             0.386, 0.942, 0.84994984, 0.95, 0.704,
                                                             0.272, 0.133, 0.14, 1., 0.294,
                                                             0.08741667, 0.00229286, 0.1719, 0.1, 0.832,
                                                             0.014, 0.012, 0.99, 0.49186667])])

        fitter.evaluate_holdout(holdout_x=test_data, holdout_y=test_data_y)
        assert fitter.holdout_evaluators[0].metric_name == 'kappa'
        assert isclose(fitter.holdout_accuracies[0], 0.581636060100167)

    def test_RandomForestMW_regression(self):
        data = TestHelper.get_cement_data()
        splitter = RegressionStratifiedDataSplitter(test_ratio=0.20)
        training_indexes, test_indexes = splitter.split(target_values=data.strength)

        train_data = data.iloc[training_indexes]
        train_data_y = train_data.strength
        train_data = train_data.drop('strength', axis=1)

        test_data = data.iloc[test_indexes]
        test_data_y = test_data.strength
        test_data = test_data.drop('strength', axis=1)

        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        transformations = [ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             evaluators=[RmseEvaluator()])
        fitter.fit(data_x=train_data, data_y=train_data_y, hyper_params=RandomForestHP(criterion='MAE',
                                                                                       n_estimators=10))
        assert fitter.model_info.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic',
                                                   'coarseagg', 'fineagg', 'age']

        assert fitter.model_info.hyper_params.params_dict == {'n_estimators': 10,
                                                              'criterion': 'mae',
                                                              'max_features': None,
                                                              'max_depth': None,
                                                              'min_samples_split': 2,
                                                              'min_samples_leaf': 1,
                                                              'min_weight_fraction_leaf': 0.0,
                                                              'max_leaf_nodes': None,
                                                              'min_impurity_decrease': 0,
                                                              'bootstrap': True,
                                                              'oob_score': False,
                                                              'n_jobs': -1,
                                                              'random_state': 42}
        assert fitter.training_evaluators[0].metric_name == Metric.ROOT_MEAN_SQUARE_ERROR.value
        assert isclose(fitter.training_accuracies[0], 2.6355960174911188)

        fitter.evaluate_holdout(holdout_x=test_data, holdout_y=test_data_y)
        assert fitter.holdout_evaluators[0].metric_name == Metric.ROOT_MEAN_SQUARE_ERROR.value
        assert isclose(fitter.holdout_accuracies[0], 5.3865850590426003)
