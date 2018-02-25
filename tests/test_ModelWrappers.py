import os
import os.path
import pickle
import shutil
import warnings
from math import isclose
from typing import Callable

import numpy as np
import pandas as pd
from mock import patch

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

    # noinspection PyArgumentList
    def play_time(self):
        ######################################################################################################
        # REGRESSION
        ######################################################################################################
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformed_data = data.drop(columns='fineagg')
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_regression(transformed_data, target_variable)  # noqa

        # data.iloc[403]
        # splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.20)
        # training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])
        # assert len(training_indexes) == 824
        # pd.Series(training_indexes).to_csv('tests/data/cement_training_indexes.csv')

        ##########################
        # Linear Regression
        ##########################
        from sklearn import linear_model
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        regr = linear_model.LinearRegression()
        regr.fit(train_x, train_y)
        # training error
        predictions = regr.predict(X=train_x)

        # noinspection PyUnboundLocalVariable
        np.sqrt(mean_squared_error(y_true=train_y, y_pred=predictions))
        assert isclose(mean_squared_error(y_true=train_y, y_pred=predictions), 109.68243774089586)
        assert isclose(mean_absolute_error(y_true=train_y, y_pred=predictions), 8.360259532214116)
        # holdout error
        predictions = regr.predict(X=holdout_x)
        assert isclose(mean_squared_error(y_true=holdout_y, y_pred=predictions), 100.07028301004217)
        assert isclose(mean_absolute_error(y_true=holdout_y, y_pred=predictions), 7.99161252047238)

        import statsmodels.api as sm
        model_object = sm.OLS(train_y, sm.add_constant(train_x)).fit()
        model_object.pvalues.values.round(5)

        ######################################################################################################
        # CLASSIFICATION
        ######################################################################################################
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'

        # data.iloc[403]
        # splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.20)
        # training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])
        # assert len(training_indexes) == 712
        # pd.Series(training_indexes).to_csv('tests/data/titanic_training_indexes.csv')

        global_transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                                  CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                                  ImputationTransformer(),
                                  DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_class(data, target_variable)
        pipeline = TransformerPipeline(transformations=global_transformations)
        train_x = pipeline.fit_transform(data_x=train_x)

        ##########################
        # Logistic Regression
        ##########################
        from sklearn import linear_model
        logistic = linear_model.LogisticRegression(random_state=42)

        logistic.fit(train_x, train_y)
        predicted_probabilities = logistic.predict_proba(X=pipeline.transform(data_x=holdout_x))

        predicted_classes_calc = [1 if x > 0.5 else 0 for x in predicted_probabilities[:, 1]]
        predicted_classes = logistic.predict(X=pipeline.transform(data_x=holdout_x))
        assert all(predicted_classes == predicted_classes_calc)

        from sklearn.metrics import roc_curve
        from sklearn.metrics import roc_auc_score
        # noinspection PyUnboundLocalVariable
        fpr, tpr, thresholds = roc_curve(y_true=holdout_y, y_score=pd.DataFrame(predicted_probabilities)[1])
        roc_auc = roc_auc_score(y_true=holdout_y, y_score=pd.DataFrame(predicted_probabilities)[1])

        # Plot ROC curve
        import matplotlib.pyplot as plt
        plt.plot(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate or (1 - Specifity)')
        plt.ylabel('True Positive Rate or (Sensitivity)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        from sklearn.metrics import precision_recall_curve
        import matplotlib.pyplot as plt

        precision, recall, _ = precision_recall_curve(y_true=holdout_y,
                                                      probas_pred=pd.DataFrame(predicted_probabilities)[1])
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall')

        ##########################
        # Multi-Class Random Forest
        ##########################
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier

        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)

        test_splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.25)
        training_indexes, test_indexes = test_splitter.split(target_values=pd.Categorical.from_codes(iris.target, iris.target_names).get_values())  # noqa

        # training_y = data.iloc[training_indexes][target_variable]
        # training_x = data.iloc[training_indexes].drop(columns=target_variable)
        #
        # holdout_y = data.iloc[holdout_indexes][target_variable]
        # holdout_x = data.iloc[holdout_indexes].drop(columns=target_variable)

        df['is_train'] = False
        df.loc[training_indexes, 'is_train'] = True
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        df.head()

        train, test = df[df['is_train'] == True], df[df['is_train'] == False]  # noqa
        len(test)

        features = df.columns[:4]
        clf = RandomForestClassifier(n_jobs=2, random_state=42)
        y, _ = pd.factorize(train['species'])
        clf.fit(train[features], y)

        clf.predict_proba(test[features])

        temp = pd.DataFrame(clf.predict_proba(test[features]))
        temp.columns = ['setosa', 'versicolor', 'virginica']
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/random_forest_multiclass_output.pkl'))  # noqa
        with open(file, 'wb') as output:
            pickle.dump(temp, output, pickle.HIGHEST_PROTOCOL)

        preds = iris.target_names[clf.predict(test[features])]
        pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])

        evaluator = MultiClassEvaluator.from_classes(actual_classes=test['species'], predicted_classes=preds)
        assert evaluator.all_quality_metrics is not None

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
        train_x, train_y, test_x, test_y = TestHelper.split_train_holdout_regression(data, target_variable)
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
        train_x, train_y, test_x, test_y = TestHelper.split_train_holdout_regression(data, target_variable)

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
        assert isclose(RmseScore().calculate(actual_values=test_y, predicted_values=predictions), 23.528246193289437)  # noqa

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
        assert isclose(RmseScore().calculate(actual_values=test_y, predicted_values=predictions), 23.528246193289437)  # noqa

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

    def test_Regression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelFitter(model=LinearRegression(),
                             model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                             splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                             evaluator=RegressionEvaluator(),
                             persistence_manager=None,
                             train_callback=None)

        fitter.fit(data=data, target_variable=target_variable, hyper_params=None)
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert isclose(fitter.training_evaluator.mean_squared_error, 109.68243774089586)
        assert isclose(fitter.training_evaluator.mean_absolute_error, 8.360259532214116)
        assert isclose(fitter.holdout_evaluator.mean_squared_error, 100.07028301004217)
        assert isclose(fitter.holdout_evaluator.mean_absolute_error, 7.99161252047238)

        assert fitter.model_info.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic',
                                                   'coarseagg', 'age']

        assert isclose(10.524151070654748, fitter.model_info.summary_stats['residual standard error (RSE)'])
        assert isclose(0.6065189841627049, fitter.model_info.summary_stats['adjusted r-squared'])
        assert isclose(4.896209721643315e-162, fitter.model_info.summary_stats['model p-value'])
        assert isclose(0.6276616298351786, fitter.model_info.summary_stats['Ratio RSE to Target STD'])
        assert isclose(0.015009316585157098, fitter.model_info.summary_stats['Residual Correlations'])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_concrete_regression_results.pkl'))  # noqa
        # SAVE THE ORIGINAL RESULTS, WHICH WERE VERIFIED AGAINST R'S LM() FUNCTION
        # with open(file, 'wb') as output:
        #     pickle.dump(fitter.model_info.results_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            regression_results_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=regression_results_summary,
                                                      data_frame2=fitter.model_info.results_summary)

        TestHelper.check_plot('data/test_ModelWrappers/test_RegressionMW_regression_plots.png',
                              lambda: fitter.model_info.graph)

    def test_ModelFitter_callback(self):
        # make sure that the ModelFitter->train_callback works, other tests rely on it to work correctly.
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        # noinspection PyUnusedLocal
        def train_callback(data_x, data_y, hyper_params):
            raise NotImplementedError()

        model_fitter = ModelFitter(model=MockRegressionModelWrapper(data_y=data.strength),
                                   model_transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),
                                                          ImputationTransformer(),
                                                          DummyEncodeTransformer()],
                                   splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                                   evaluator=RegressionEvaluator(),
                                   train_callback=train_callback)

        # should raise an error from the callback definition above 
        self.assertRaises(NotImplementedError, lambda: model_fitter.fit(data=data, target_variable=target_variable, hyper_params=None))  # noqa

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
        train_x, train_y, test_x, test_y = TestHelper.split_train_holdout_regression(data, target_variable)
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
        # make sure the data that we pass to `train()` in the ModelWrapper is transformed
        # then make sure what we get in the callback matches the transformed data
        ######################################################################################################
        test_pipeline = TransformerPipeline(transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),  # noqa
                                                             ImputationTransformer(),
                                                             DummyEncodeTransformer()])
        transformed_data = test_pipeline.fit_transform(data_x=train_x)
        # make sure our test transformations are transformed as expected (although this should already be
        # tested in test_Transformations file
        assert all(transformed_data.columns.values == ['cement', 'slag', 'ash', 'water', 'superplastic', 'age', 'random_code1'])  # noqa
        assert OOLearningHelpers.is_series_numeric(variable=transformed_data.random_code1)
        assert transformed_data.isna().sum().sum() == 0

        # this callback will be called by the ModelWrapper before fitting the model
        # the callback gives us back the data that it will pass to the underlying model
        # so we can make sure it matches what we expect
        def train_callback(data_x, data_y, hyper_params):
            assert hyper_params is None
            assert len(data_y) == len(train_y)
            assert all(data_y == train_y)
            TestHelper.ensure_all_values_equal(data_frame1=transformed_data, data_frame2=data_x)

        ######################################################################################################
        # fit/predict the model using the Mock object, which stores the transformed training/test data
        # so we can validate the expected transformations took place across both datasets
        ######################################################################################################
        model_fitter = ModelFitter(model=MockRegressionModelWrapper(data_y=data.strength),
                                   model_transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),
                                                          ImputationTransformer(),
                                                          DummyEncodeTransformer()],
                                   evaluator=RegressionEvaluator(),
                                   splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                                   train_callback=train_callback)

        # should raise an error calling `predict` before `fit`
        self.assertRaises(ModelNotFittedError, lambda: model_fitter.predict(data_x=test_x))

        model_fitter.fit(data=data, target_variable=target_variable, hyper_params=None)

        assert isclose(model_fitter.training_evaluator.root_mean_squared_error, 24.50861705505752)
        assert isclose(model_fitter.training_evaluator.mean_absolute_error, 19.700946601941748)

        assert isclose(model_fitter.holdout_evaluator.root_mean_squared_error, 23.528246193289437)
        assert isclose(model_fitter.holdout_evaluator.mean_absolute_error, 19.254368932038837)

        # should not be able to call fit twice
        self.assertRaises(ModelAlreadyFittedError, lambda: model_fitter.fit(data=data,
                                                                            target_variable=target_variable,
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

    def test_Logistic(self):
        warnings.filterwarnings("ignore")
        # noinspection PyUnusedLocal
        with patch('sys.stdout', new=MockDevice()) as fake_out:  # supress output of logistic model
            data = TestHelper.get_titanic_data()
            transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                               CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                               ImputationTransformer(),
                               DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

            # test with custom threshold of 0.5
            fitter = ModelFitter(model=LogisticRegression(),
                                 model_transformations=transformations,
                                 splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                 evaluator=TwoClassProbabilityEvaluator(
                                     converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)))  # noqa
            fitter.fit(data=data, target_variable='Survived')
            assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
            assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

            con_matrix = fitter.training_evaluator._confusion_matrix
            assert con_matrix.matrix.loc[:, 0].values.tolist() == [386, 85, 471]
            assert con_matrix.matrix.loc[:, 1].values.tolist() == [53, 188, 241]
            assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [439, 273, 712]
            assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
            assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
            assert isclose(fitter.training_evaluator.auc, 0.860346942351498)
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_training_ROC.png',
                                  lambda: fitter.training_evaluator.get_roc_curve())
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_training_PrecRecal.png',
                                  lambda: fitter.training_evaluator.get_ppv_tpr_curve())

            con_matrix = fitter.holdout_evaluator._confusion_matrix
            assert con_matrix.matrix.loc[:, 0].values.tolist() == [98, 22, 120]
            assert con_matrix.matrix.loc[:, 1].values.tolist() == [12, 47, 59]
            assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [110, 69, 179]
            assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
            assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
            assert isclose(fitter.holdout_evaluator.auc, 0.8454545454545455)
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_holdout_ROC.png',
                                  lambda: fitter.holdout_evaluator.get_roc_curve())
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_holdout_PrecRecal.png',
                                  lambda: fitter.holdout_evaluator.get_ppv_tpr_curve())

            actual_metrics = fitter.training_evaluator.all_quality_metrics
            expected_metrics = {'Kappa': 0.5807869204973077, 'F1 Score': 0.7315175097276264, 'Two-Class Accuracy': 0.8061797752808989, 'Error Rate': 0.19382022471910113, 'True Positive Rate': 0.6886446886446886, 'True Negative Rate': 0.8792710706150342, 'False Positive Rate': 0.12072892938496584, 'False Negative Rate': 0.31135531135531136, 'Positive Predictive Value': 0.7800829875518672, 'Negative Predictive Value': 0.8195329087048833, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
            assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
            assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

            actual_metrics = fitter.holdout_evaluator.all_quality_metrics
            expected_metrics = {'Kappa': 0.5879485443466486, 'F1 Score': 0.7343750000000001, 'Two-Class Accuracy': 0.8100558659217877, 'Error Rate': 0.18994413407821228, 'True Positive Rate': 0.6811594202898551, 'True Negative Rate': 0.8909090909090909, 'False Positive Rate': 0.10909090909090909, 'False Negative Rate': 0.3188405797101449, 'Positive Predictive Value': 0.7966101694915254, 'Negative Predictive Value': 0.8166666666666667, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
            assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
            assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

    def test_LogisticMW_string_target(self):
        data = TestHelper.get_titanic_data()
        positive_class = 'lived'
        negative_class = 'died'
        # Test with a string target variable rather than 0/1
        data.Survived = np.where(data.Survived == 1, positive_class, negative_class)
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

        # test with custom threshold of 0.5
        fitter = ModelFitter(model=LogisticRegression(),
                             model_transformations=transformations,
                             splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                             evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=positive_class)))
        fitter.fit(data=data, target_variable='Survived')
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        con_matrix = fitter.training_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 'died'].values.tolist() == [386, 85, 471]
        assert con_matrix.matrix.loc[:, 'lived'].values.tolist() == [53, 188, 241]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [439, 273, 712]
        assert con_matrix.matrix.index.values.tolist() == ['died', 'lived', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['died', 'lived', 'Total']
        assert isclose(fitter.training_evaluator.auc, 0.860346942351498)

        con_matrix = fitter.holdout_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 'died'].values.tolist() == [98, 22, 120]
        assert con_matrix.matrix.loc[:, 'lived'].values.tolist() == [12, 47, 59]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [110, 69, 179]
        assert con_matrix.matrix.index.values.tolist() == ['died', 'lived', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['died', 'lived', 'Total']
        assert isclose(fitter.holdout_evaluator.auc, 0.8454545454545455)

        actual_metrics = fitter.training_evaluator.all_quality_metrics
        expected_metrics = {'Kappa': 0.5807869204973077, 'F1 Score': 0.7315175097276264, 'Two-Class Accuracy': 0.8061797752808989, 'Error Rate': 0.19382022471910113, 'True Positive Rate': 0.6886446886446886, 'True Negative Rate': 0.8792710706150342, 'False Positive Rate': 0.12072892938496584, 'False Negative Rate': 0.31135531135531136, 'Positive Predictive Value': 0.7800829875518672, 'Negative Predictive Value': 0.8195329087048833, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

        actual_metrics = fitter.holdout_evaluator.all_quality_metrics
        expected_metrics = {'Kappa': 0.5879485443466486, 'F1 Score': 0.7343750000000001, 'Two-Class Accuracy': 0.8100558659217877, 'Error Rate': 0.18994413407821228, 'True Positive Rate': 0.6811594202898551, 'True Negative Rate': 0.8909090909090909, 'False Positive Rate': 0.10909090909090909, 'False Negative Rate': 0.3188405797101449, 'Positive Predictive Value': 0.7966101694915254, 'Negative Predictive Value': 0.8166666666666667, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

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
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        cache_directory = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_RandomForestMW_classification')  # noqa
        # test with custom threshold of 0.5
        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                             evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)),
                             persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        assert fitter._persistence_manager._cache_directory == cache_directory
        shutil.rmtree(fitter._persistence_manager._cache_directory)
        assert not os.path.isdir(fitter._persistence_manager._cache_directory)
        fitter.fit(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini'))
        assert os.path.isfile(fitter._persistence_manager._cache_path)

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

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

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 0.9642058165548097, 'F1 Score': 0.9777777777777779, 'Two-Class Accuracy': 0.9831460674157303, 'Error Rate': 0.016853932584269662, 'True Positive Rate': 0.967032967032967, 'True Negative Rate': 0.9931662870159453, 'False Positive Rate': 0.00683371298405467, 'False Negative Rate': 0.03296703296703297, 'Positive Predictive Value': 0.9887640449438202, 'Negative Predictive Value': 0.9797752808988764, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.581636060100167, 'F1 Score': 0.736842105263158, 'Two-Class Accuracy': 0.8044692737430168, 'Error Rate': 0.19553072625698323, 'True Positive Rate': 0.7101449275362319, 'True Negative Rate': 0.8636363636363636, 'False Positive Rate': 0.13636363636363635, 'False Negative Rate': 0.2898550724637681, 'Positive Predictive Value': 0.765625, 'Negative Predictive Value': 0.8260869565217391, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

        ######################################################################################################
        # test custom hyper-parameters
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        cache_directory = TestHelper.ensure_test_directory(
            'data/test_ModelWrappers/cached_test_models/test_RandomForestMW_classification')  # noqa
        # test with custom threshold of 0.5
        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                             evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)),
                             persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        assert fitter._persistence_manager._cache_directory == cache_directory
        shutil.rmtree(fitter._persistence_manager._cache_directory)
        assert not os.path.isdir(fitter._persistence_manager._cache_directory)
        fitter.fit(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini',
                                                                                      max_features='auto',
                                                                                      n_estimators=10,
                                                                                      n_jobs=2))
        assert os.path.isfile(fitter._persistence_manager._cache_path)

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model_info.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3',
                                                   'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2',
                                                   'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0',
                                                   'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5',
                                                   'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
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

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 0.9491855583543242, 'F1 Score': 0.9683426443202979, 'Two-Class Accuracy': 0.976123595505618, 'Error Rate': 0.023876404494382022, 'True Positive Rate': 0.9523809523809523, 'True Negative Rate': 0.9908883826879271, 'False Positive Rate': 0.009111617312072893, 'False Negative Rate': 0.047619047619047616, 'Positive Predictive Value': 0.9848484848484849, 'Negative Predictive Value': 0.9709821428571429, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.5503428610224728, 'F1 Score': 0.7086614173228347, 'Two-Class Accuracy': 0.7932960893854749, 'Error Rate': 0.20670391061452514, 'True Positive Rate': 0.6521739130434783, 'True Negative Rate': 0.8818181818181818, 'False Positive Rate': 0.11818181818181818, 'False Negative Rate': 0.34782608695652173, 'Positive Predictive Value': 0.7758620689655172, 'Negative Predictive Value': 0.8016528925619835, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

    def test_RandomForestMW_classification_string_target(self):
        data = TestHelper.get_titanic_data()
        positive_class = 'lived'
        negative_class = 'died'

        # Test with a string target variable rather than 0/1
        data.Survived = np.where(data.Survived == 1, positive_class, negative_class)

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        # test with custom threshold of 0.5
        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                             evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=positive_class)))

        fitter.fit(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini'))

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

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

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 0.9642058165548097, 'F1 Score': 0.9777777777777779, 'Two-Class Accuracy': 0.9831460674157303, 'Error Rate': 0.016853932584269662, 'True Positive Rate': 0.967032967032967, 'True Negative Rate': 0.9931662870159453, 'False Positive Rate': 0.00683371298405467, 'False Negative Rate': 0.03296703296703297, 'Positive Predictive Value': 0.9887640449438202, 'Negative Predictive Value': 0.9797752808988764, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.581636060100167, 'F1 Score': 0.736842105263158, 'Two-Class Accuracy': 0.8044692737430168, 'Error Rate': 0.19553072625698323, 'True Positive Rate': 0.7101449275362319, 'True Negative Rate': 0.8636363636363636, 'False Positive Rate': 0.13636363636363635, 'False Negative Rate': 0.2898550724637681, 'Positive Predictive Value': 0.765625, 'Negative Predictive Value': 0.8260869565217391, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

        con_matrix = fitter.training_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 'died'].values.tolist() == [436, 9, 445]
        assert con_matrix.matrix.loc[:, 'lived'].values.tolist() == [3, 264, 267]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [439, 273, 712]
        assert con_matrix.matrix.index.values.tolist() == ['died', 'lived', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['died', 'lived', 'Total']

        con_matrix = fitter.holdout_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 'died'].values.tolist() == [95, 20, 115]
        assert con_matrix.matrix.loc[:, 'lived'].values.tolist() == [15, 49, 64]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [110, 69, 179]
        assert con_matrix.matrix.index.values.tolist() == ['died', 'lived', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['died', 'lived', 'Total']

    def test_RandomForestMW_regression(self):
        data = TestHelper.get_cement_data()
        transformations = [ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=transformations,
                             splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                             evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        fitter.fit(data=data, target_variable='strength', hyper_params=RandomForestHP(criterion='MAE', n_estimators=10))  # noqa

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

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

        keys = fitter.training_evaluator.all_quality_metrics.keys()
        expected_values = {'Mean Absolute Error (MAE)': 1.6050794902912628, 'Mean Squared Error (MSE)': 6.946366367415049, 'Root Mean Squared Error (RMSE)': 2.6355960174911193, 'RMSE to Standard Deviation of Target': 0.15718726202422936}  # noqa
        assert all([isclose(fitter.training_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

        expected_values = {'Mean Absolute Error (MAE)': 3.7880849514563115, 'Mean Squared Error (MSE)': 29.015298598300976, 'Root Mean Squared Error (RMSE)': 5.3865850590426, 'RMSE to Standard Deviation of Target': 0.3281385595158624}  # noqa
        assert all([isclose(fitter.holdout_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

    def test_RandomForestMW_classification_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelFitter(model=RandomForestMW(),
                             model_transformations=None,
                             splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                             evaluator=MultiClassEvaluator(converter=HighestValueConverter()))

        fitter.fit(data=data, target_variable=target_variable, hyper_params=RandomForestHP(criterion='gini',
                                                                                           n_estimators=10,
                                                                                           max_features='auto'))  # noqa

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model_info.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # noqa
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
                                                              'n_jobs': -1,
                                                              'random_state': 42}

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 1.0, 'Accuracy': 1.0, 'Error Rate': 0.0, 'No Information Rate': 0.3392857142857143, 'Total Observations': 112}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.841995841995842, 'Accuracy': 0.8947368421052632, 'Error Rate': 0.10526315789473684, 'No Information Rate': 0.34210526315789475, 'Total Observations': 38}  # noqa

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix['versicolor'].values.tolist() == [0, 12, 3, 15]
        assert con_matrix['virginica'].values.tolist() == [0, 1, 10, 11]
        assert con_matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
