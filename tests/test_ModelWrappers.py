import copy
import os
import os.path
import pickle
import shutil
import time
from math import isclose
from typing import Callable

import numpy as np
import pandas as pd

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    import statsmodels.api as sm  # https://github.com/statsmodels/statsmodels/issues/3814

from sklearn.metrics import roc_auc_score  # noqa
# noinspection PyProtectedMember
from sklearn.utils import shuffle  # noqa

from mock import patch  # noqa
from oolearning import *  # noqa
from tests.MockClassificationModelWrapper import MockClassificationModelWrapper  # noqa
from tests.MockRegressionModelWrapper import MockRegressionModelWrapper  # noqa
from tests.TestHelper import TestHelper  # noqa
from tests.TimerTestCase import TimerTestCase  # noqa


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

    def set_sub_structure(self, sub_structure: str):
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

        ######################################################################################################
        # playing with sklearn ModelAggregator
        ######################################################################################################
        from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
        from sklearn.ensemble import AdaBoostClassifier as SkAdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier as SkDecisionTreeClassifier
        from sklearn.ensemble import VotingClassifier as SkVotingClassifier

        clf1 = SkRandomForestClassifier(random_state=42)
        clf1 = clf1.fit(train_x, train_y)
        # clf1.predict_proba(holdout_x)
        roc_auc_score(y_true=holdout_y, y_score=clf1.predict_proba(holdout_x)[:, 1])

        clf2 = SkAdaBoostClassifier(random_state=42)
        clf2 = clf2.fit(train_x, train_y)
        # clf2.predict_proba(holdout_x)
        roc_auc_score(y_true=holdout_y, y_score=clf2.predict_proba(holdout_x)[:, 1])

        clf3 = SkDecisionTreeClassifier(random_state=42)
        clf3 = clf3.fit(train_x, train_y)
        # clf3.predict_proba(holdout_x)
        roc_auc_score(y_true=holdout_y, y_score=clf3.predict_proba(holdout_x)[:, 1])

        eclf1 = SkVotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
        eclf1 = eclf1.fit(train_x, train_y)
        eclf1.predict(holdout_x)
        roc_auc_score(y_true=holdout_y, y_score=eclf1.predict_proba(holdout_x)[:, 1])

        eclf2 = SkVotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
        eclf2 = eclf2.fit(train_x, train_y)
        # eclf2.predict_proba(holdout_x)
        roc_auc_score(y_true=holdout_y, y_score=eclf2.predict_proba(holdout_x)[:, 1])

        eclf3 = SkVotingClassifier(estimators=[
            ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
            voting='soft', weights=[2, 1, 1],
            flatten_transform=True)
        eclf3 = eclf3.fit(X, y)
        print(eclf3.predict(X))

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

        assert TestHelper.ensure_all_values_equal(data_frame1=data.head(n=30),
                                                  data_frame2=mock_model.data_x_trained_head)

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

    def test_ModelWrapperBase_regressor(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        train_x, train_y, test_x, test_y = TestHelper.split_train_holdout_regression(data, target_variable)
        ######################################################################################################
        # test predicting without training, training an already trained model, properties without training
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)

        # should raise an error if only one tuning_parameters is passed in, since expecting 2 params
        self.assertRaises(ModelNotFittedError,
                          lambda: model_wrapper.predict(data_x=train_x))

        self.assertRaises(ModelNotFittedError, lambda: model_wrapper.model_object)
        self.assertRaises(ModelNotFittedError, lambda: model_wrapper.feature_names)
        self.assertRaises(ModelNotFittedError, lambda: model_wrapper.hyper_params)

        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        assert model_wrapper.results_summary == 'test_summary'
        assert model_wrapper.hyper_params.test == 'test hyper-params'

        self.assertRaises(ModelAlreadyFittedError,
                          lambda: model_wrapper.train(data_x=train_x,
                                                      data_y=train_y,
                                                      hyper_params=MockHyperParams()))

        predictions = model_wrapper.predict(data_x=train_x)
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)

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

        if os.path.isdir(cache_directory) or os.path.isfile(file_path):
            shutil.rmtree(cache_directory)

        assert os.path.isdir(cache_directory) is False
        assert os.path.isfile(file_path) is False

        model_wrapper.set_persistence_manager(persistence_manager=LocalCacheManager(cache_directory=cache_directory, key=cache_key))  # noqa
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        # ensure the model "trained"
        assert model_wrapper.results_summary == 'test_summary'
        assert model_wrapper.hyper_params.test == 'test hyper-params'
        # ensure the model is now cached
        assert os.path.isfile(file_path) is True
        with open(file_path, 'rb') as saved_object:
            model_object = pickle.load(saved_object)
            # this is from the MockRegressionModelWrapper
            assert model_object._model_object == 'test model_object'

        ######################################################################################################
        # caching and it already exists
        # setting `model_object` on a cached/existing model, should not be updated in the model or the cache
        ######################################################################################################
        # first ensure that setting `model_object` results in model_object being changed
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength, model_object='new model object!!')
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        assert model_wrapper.model_object._model_object == 'new model object!!'

        # now, if we pass in the same `model_object` to a previously cached model, we should get the old value
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength, model_object='new model object!!')
        assert os.path.isfile(file_path) is True  # should already exist from above
        model_wrapper.set_persistence_manager(persistence_manager=LocalCacheManager(cache_directory=cache_directory, key=cache_key))  # noqa
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        # ensure the cached value in model_object is the same (and not changed to 'new model object!!')
        assert model_object._model_object == 'test model_object'  # CACHED value !!!!!
        # ensure the model "trained"
        assert model_wrapper.results_summary == 'test_summary'
        assert model_wrapper.hyper_params.test == 'test hyper-params'
        assert os.path.isfile(file_path) is True
        # ensure same cache (i.e. has old/cached model_object value)
        with open(file_path, 'rb') as saved_object:
            model_object = pickle.load(saved_object)
            # old model_object value ensures same cache
            assert model_object._model_object == 'test model_object'

        os.remove(file_path)  # clean up

        ######################################################################################################
        # predicting with a cached model that does not exist (need to call `train()` before `predict()`)
        # `predict()` should not change, basically testing that we have a model via model_object
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
        # because train has parameters that are needed to set additional info after model has been fitted.
        # `predict()` should not change, basically testing that we have a model via model_object
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

        ######################################################################################################
        # test sub-structure i.e. sub-directories
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)
        cache_directory = TestHelper.ensure_test_directory('data/temp_caching_tests')
        cache_key = 'test_caching_file'
        prefix = 'prefix_'
        sub_directory = 'sub'

        file_path = os.path.join(cache_directory, sub_directory, prefix + cache_key + '.pkl')

        if os.path.isdir(cache_directory) or os.path.isfile(file_path):
            shutil.rmtree(cache_directory)
        assert os.path.isdir(cache_directory) is False
        assert os.path.isfile(file_path) is False

        model_wrapper.set_persistence_manager(
            persistence_manager=LocalCacheManager(cache_directory=cache_directory,
                                                  sub_directory=sub_directory,
                                                  key=cache_key,
                                                  key_prefix=prefix))

        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        # ensure the model "trained"
        assert model_wrapper.results_summary == 'test_summary'
        assert model_wrapper.hyper_params.test == 'test hyper-params'
        # ensure the model is now cached
        assert os.path.isfile(file_path) is True
        with open(file_path, 'rb') as saved_object:
            model_object = pickle.load(saved_object)
            # this is from the MockRegressionModelWrapper
            assert model_object._model_object == 'test model_object'
        shutil.rmtree(cache_directory)
        ######################################################################################################
        # test sub-structure i.e. sub-directories vis `set_substructure after initializing
        ######################################################################################################
        model_wrapper = MockRegressionModelWrapper(data_y=data.strength)
        cache_directory = TestHelper.ensure_test_directory('data/temp_caching_tests')
        cache_key = 'test_caching_file'
        prefix = 'prefix_'
        sub_directory = 'sub'

        file_path = os.path.join(cache_directory, sub_directory, prefix + cache_key + '.pkl')

        if os.path.isdir(cache_directory) or os.path.isfile(file_path):
            shutil.rmtree(cache_directory)
        assert os.path.isdir(cache_directory) is False
        assert os.path.isfile(file_path) is False

        model_wrapper.set_persistence_manager(
            persistence_manager=LocalCacheManager(cache_directory=cache_directory,
                                                  # do NOT set sub_directory
                                                  sub_directory=None,
                                                  key=cache_key,
                                                  key_prefix=prefix))
        # no sub-directory
        TestHelper.ensure_test_directory('data/temp_caching_tests')
        assert model_wrapper._persistence_manager._cache_directory == TestHelper.ensure_test_directory('data/temp_caching_tests')  # noqa
        assert model_wrapper._persistence_manager._cache_path == TestHelper.ensure_test_directory('data/temp_caching_tests/prefix_test_caching_file.pkl')  # noqa
        # set sub_structure via setter
        model_wrapper._persistence_manager.set_sub_structure(sub_structure=sub_directory)
        # now should contain sub-directory
        assert model_wrapper._persistence_manager._cache_directory == TestHelper.ensure_test_directory('data/temp_caching_tests/sub')  # noqa
        assert model_wrapper._persistence_manager._cache_path == TestHelper.ensure_test_directory('data/temp_caching_tests/sub/prefix_test_caching_file.pkl')  # noqa
        model_wrapper.train(data_x=train_x, data_y=train_y, hyper_params=MockHyperParams())
        # ensure the model "trained"
        assert model_wrapper.results_summary == 'test_summary'
        assert model_wrapper.hyper_params.test == 'test hyper-params'
        # ensure the model is now cached
        assert os.path.isfile(file_path) is True
        with open(file_path, 'rb') as saved_object:
            model_object = pickle.load(saved_object)
            # this is from the MockRegressionModelWrapper
            assert model_object._model_object == 'test model_object'
        shutil.rmtree(cache_directory)

    def test_HyperParamsBase(self):
        params = MockHyperParams()
        assert params.params_dict == dict(a='a', b='b', c='c')
        params.update_dict(dict(b=1, c=None))
        assert params.params_dict == dict(a='a', b=1, c=None)
        params.update_dict(dict(b=1, c=None))

        # cannot update non-existant hyper-param (d)
        self.assertRaises(ValueError, lambda: params.update_dict(dict(d='d')))

    def test_ModelTrainer_expected_columns_after_dummy_transform_with_missing_categories(self):
        # before we fit the data, we actually want to 'snoop' at what the expected columns will be with
        # ALL the data. The reason is that if we so some sort of dummy encoding, but not all the
        # categories are included in the training set (i.e. maybe only a small number of observations have
        # the categoric value), then we can still ensure that we will be giving the same expected columns/
        # encodings to the predict method with the holdout set.
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # splitting with a very high holdout ratio means we will miss some categories in the training set that
        # might (in this case, will be) found in the holdout set.
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.9)
        training_indexes, _ = splitter.split(data[target_variable])
        training_data = data.iloc[training_indexes].drop(columns=target_variable)
        training_columns_original = TransformerPipeline.get_expected_columns(data=training_data, transformations=transformations)  # noqa
        expected_columns = TransformerPipeline.get_expected_columns(data=data.drop(columns=target_variable), transformations=transformations)  # noqa
        # these are the categories that will not be found in the training set;
        # therefore, there wouldn't have been any associated columns
        expected_missing = ['SibSp_5', 'Parch_3', 'Parch_6']
        assert all([x not in training_columns_original for x in expected_missing])  # not in training set
        assert all([x in expected_columns for x in expected_missing])  # in expected columns

        # can use the callback which returns the transformed training data
        # we know that we would have had columns missing, so verify they aren't
        # noinspection PyUnusedLocal
        def train_callback(transformed_training_data, data_y, hyper_params):
            # verify columns exist and for each (otherwise) missing column, all values are 0
            assert all(transformed_training_data.columns.values == expected_columns)
            assert all(transformed_training_data['SibSp_5'] == 0)
            assert all(transformed_training_data['Parch_3'] == 0)
            assert all(transformed_training_data['Parch_6'] == 0)

        # same holdout_ratio as above
        fitter = ModelTrainer(model=RandomForestClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.9),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)),
                              train_callback=train_callback)
        fitter.train(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini'))

    def test_Regression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelTrainer(model=LinearRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=None)

        fitter.train(data=data, target_variable=target_variable, hyper_params=None)
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert isclose(fitter.training_evaluator.mean_squared_error, 109.68243774089586)
        assert isclose(fitter.training_evaluator.mean_absolute_error, 8.360259532214116)
        assert isclose(fitter.holdout_evaluator.mean_squared_error, 100.07028301004217)
        assert isclose(fitter.holdout_evaluator.mean_absolute_error, 7.99161252047238)

        np.asarray(fitter.predict(data_x=data.drop(columns=target_variable)))

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'age']  # noqa

        assert isclose(10.524151070654748, fitter.model.summary_stats['residual standard error (RSE)'])
        assert isclose(0.6065189841627049, fitter.model.summary_stats['adjusted r-squared'])
        assert isclose(4.896209721643315e-162, fitter.model.summary_stats['model p-value'])
        assert isclose(0.6276616298351786, fitter.model.summary_stats['Ratio RSE to Target STD'])
        assert isclose(0.015009316585157098, fitter.model.summary_stats['Residual Correlations'])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_RegressionMWTest_concrete_regression_results.pkl'))  # noqa
        # SAVE THE ORIGINAL RESULTS, WHICH WERE VERIFIED AGAINST R'S LM() FUNCTION
        # with open(file, 'wb') as output:
        #     pickle.dump(fitter.model.results_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            regression_results_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=regression_results_summary,
                                                      data_frame2=fitter.model.results_summary)

        TestHelper.check_plot('data/test_ModelWrappers/test_RegressionMW_regression_plots.png',
                              lambda: fitter.model.graph)

    def test_RidgeRegression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelTrainer(model=RidgeRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=None)

        fitter.train(data=data, target_variable=target_variable, hyper_params=RidgeRegressorHP(alpha=0))
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert fitter.model.hyper_params.params_dict == {'alpha': 0, 'solver': 'cholesky'}
        # alpha of 0 should be the same as plain Linear Regression (values are copied from above
        assert isclose(fitter.training_evaluator.mean_squared_error, 109.68243774089586)
        assert isclose(fitter.training_evaluator.mean_absolute_error, 8.360259532214116)
        assert isclose(fitter.holdout_evaluator.mean_squared_error, 100.07028301004217)
        assert isclose(fitter.holdout_evaluator.mean_absolute_error, 7.99161252047238)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'age']  # noqa

        # Test that tuner works with hyper-params
        # I also ran a Resampler with LinearRegressor model and made sure it had the same values as
        # as the Tuner.Resampler with Alpha == 0
        train_data_y = data[target_variable]
        train_data = data.drop(columns=target_variable)
        evaluators = [MaeScore(), RmseScore()]
        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=RidgeRegressor(),
                                                                      transformations=[RemoveColumnsTransformer(columns=['fineagg']),  # noqa
                                                                                       CenterScaleTransformer()],  # noqa
                                                                      scores=evaluators),
                           hyper_param_object=RidgeRegressorHP())
        grid = HyperParamsGrid(params_dict={'alpha': [0, 0.5, 1]})
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)
        assert len(tuner.results._tune_results_objects) == 3
        assert tuner.results.num_param_combos == 3
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Ridge_can_tune.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            saved_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=saved_results.resampled_stats,
                                                      data_frame2=tuner.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=saved_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

    def test_LassoRegression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelTrainer(model=LassoRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=None)

        fitter.train(data=data, target_variable=target_variable, hyper_params=LassoRegressorHP(alpha=0))
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert fitter.model.hyper_params.params_dict == {'alpha': 0}
        # alpha of 0 should be the same as plain Linear Regression (values are copied from above
        assert isclose(fitter.training_evaluator.mean_squared_error, 109.68243774089586)
        assert isclose(fitter.training_evaluator.mean_absolute_error, 8.360259532214116)
        assert isclose(fitter.holdout_evaluator.mean_squared_error, 100.07028301004217)
        assert isclose(fitter.holdout_evaluator.mean_absolute_error, 7.99161252047238)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'age']  # noqa

        # Test that tuner works with hyper-params
        # I also ran a Resampler with LinearRegressor model and made sure it had the same values as
        # as the Tuner.Resampler with Alpha == 0
        train_data_y = data[target_variable]
        train_data = data.drop(columns=target_variable)
        evaluators = [MaeScore(), RmseScore()]
        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=LassoRegressor(),
                                                                      transformations=[RemoveColumnsTransformer(columns=['fineagg']),  # noqa
                                                                                       CenterScaleTransformer()],  # noqa
                                                                      scores=evaluators),
                           hyper_param_object=LassoRegressorHP())
        grid = HyperParamsGrid(params_dict={'alpha': [0.1, 0.5, 1]})
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)
        assert len(tuner.results._tune_results_objects) == 3
        assert tuner.results.num_param_combos == 3
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Lasso_can_tune.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            saved_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=saved_results.resampled_stats,
                                                      data_frame2=tuner.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=saved_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

    def test_ElasticNetRegressor(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelTrainer(model=ElasticNetRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=None)

        fitter.train(data=data, target_variable=target_variable, hyper_params=ElasticNetRegressorHP(alpha=0))
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert fitter.model.hyper_params.params_dict == {'alpha': 0, 'l1_ratio': 0.5}

        # alpha of 0 should be the same as plain Linear Regression (values are copied from above
        assert isclose(fitter.training_evaluator.mean_squared_error, 109.68243774089586)
        assert isclose(fitter.training_evaluator.mean_absolute_error, 8.360259532214116)
        assert isclose(fitter.holdout_evaluator.mean_squared_error, 100.07028301004217)
        assert isclose(fitter.holdout_evaluator.mean_absolute_error, 7.99161252047238)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'age']  # noqa

        # Test that tuner works with hyper-params
        # I also ran a Resampler with LinearRegressor model and made sure it had the same values as
        # as the Tuner.Resampler with Alpha == 0
        train_data_y = data[target_variable]
        train_data = data.drop(columns=target_variable)
        evaluators = [MaeScore(), RmseScore()]
        tuner = ModelTuner(resampler=RepeatedCrossValidationResampler(model=ElasticNetRegressor(),
                                                                      transformations=[RemoveColumnsTransformer(columns=['fineagg']),  # noqa
                                                                                       CenterScaleTransformer()],  # noqa
                                                                      scores=evaluators),
                           hyper_param_object=ElasticNetRegressorHP())
        grid = HyperParamsGrid(params_dict={'alpha': [0.1, 0.5, 1], 'l1_ratio': [0.2, 0.6]})
        tuner.tune(data_x=train_data, data_y=train_data_y, params_grid=grid)
        assert len(tuner.results._tune_results_objects) == 6
        assert tuner.results.num_param_combos == 6
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ElasticNet_can_tune.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            saved_results = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=saved_results.resampled_stats,
                                                      data_frame2=tuner.results.resampled_stats)
            assert TestHelper.ensure_all_values_equal(data_frame1=saved_results.sorted_best_models,
                                                      data_frame2=tuner.results.sorted_best_models)

    def test_ModelFitter_callback(self):
        # make sure that the ModelTrainer->train_callback works, other tests rely on it to work correctly.
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        # noinspection PyUnusedLocal
        def train_callback(data_x, data_y, hyper_params):
            raise NotImplementedError()

        model_fitter = ModelTrainer(model=MockRegressionModelWrapper(data_y=data.strength),
                                    model_transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),
                                                           ImputationTransformer(),
                                                           DummyEncodeTransformer()],
                                    splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                                    evaluator=RegressionEvaluator(),
                                    train_callback=train_callback)

        # should raise an error from the callback definition above 
        self.assertRaises(NotImplementedError, lambda: model_fitter.train(data=data, target_variable=target_variable, hyper_params=None))  # noqa

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
        # train/predict the model using the Mock object, which stores the transformed training/test data
        # so we can validate the expected transformations took place across both datasets
        ######################################################################################################
        model_fitter = ModelTrainer(model=MockRegressionModelWrapper(data_y=data.strength),
                                    model_transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),
                                                           ImputationTransformer(),
                                                           DummyEncodeTransformer()],
                                    evaluator=RegressionEvaluator(),
                                    splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                                    train_callback=train_callback)

        # should raise an error calling `predict` before `train`
        self.assertRaises(ModelNotFittedError, lambda: model_fitter.predict(data_x=test_x))

        model_fitter.train(data=data, target_variable=target_variable, hyper_params=None)

        assert isclose(model_fitter.training_evaluator.root_mean_squared_error, 24.50861705505752)
        assert isclose(model_fitter.training_evaluator.mean_absolute_error, 19.700946601941748)

        assert isclose(model_fitter.holdout_evaluator.root_mean_squared_error, 23.528246193289437)
        assert isclose(model_fitter.holdout_evaluator.mean_absolute_error, 19.254368932038837)

        # should not be able to call train twice
        self.assertRaises(ModelAlreadyFittedError, lambda: model_fitter.train(data=data,
                                                                              target_variable=target_variable,
                                                                              hyper_params=None))

        predictions = model_fitter.predict(data_x=test_x)  # mock object stores transformed data
        assert predictions is not None
        ######################################################################################################
        # removed coarseagg and fineagg, added a categorical column and used DUMMY encoding
        ######################################################################################################
        assert model_fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'age', 'random_code1']  # noqa
        ######################################################################################################
        # ensure that we imputed the correct values in the correct indexes
        ######################################################################################################
        # training set, the Mock model wrapper saves the training data in `fitted_train_x` field, and test_x
        # so we can 'peak' inside and see the transformations

        # ensure transformation states are set correctly
        assert model_fitter._model_transformations[0].state == {}
        assert model_fitter._model_transformations[1].state == \
            {'cement': 266.19999999999999,
             'slag': 26.0,
             'ash': 0.0,
             'water': 185.69999999999999,
             'superplastic': 6.4000000000000004,
             'age': 28.0,
             'random': 'code1'}
        assert model_fitter._model_transformations[2].state == {'random': ['code0', 'code1']}

        # ensure the data is updated/imputed correctly
        assert all(model_fitter._model.fitted_train_x.loc[index_missing_train_cement]['cement'] ==
                   expected_cement_median)
        assert all(model_fitter._model.fitted_train_x.loc[index_missing_train_ash]['ash'] ==
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

        ######################################################################################################
        # ensure that we didn't change any of the original datasets
        ######################################################################################################
        assert all(train_x.loc[index_missing_train_cement]['cement'].isnull())
        assert all(train_x.loc[index_missing_train_ash]['ash'].isnull())
        assert all(test_x.loc[index_missing_test_cement]['cement'].isnull())
        assert all(test_x.loc[index_missing_test_ash]['ash'].isnull())

    def test_ModelFitter_no_splitter(self):
        # https://github.com/shane-kercheval/oo-learning/issues/3
        # "modify ModelTrainer to *optionally* take a Splitter, and train on all data"
        data = TestHelper.get_cement_data()
        original_indexes = set(data.index.values)
        # i want to shuff so i know, in my callback below, i'm actually getting back the same indices,
        # not just the fact that the indices go from 1 to x in both situations
        data = shuffle(data, random_state=42)
        assert set(data.index.values) == original_indexes  # ensure this retains all indexes
        target_variable = 'strength'

        train_x, train_y, test_x, test_y = TestHelper.split_train_holdout_regression(data, target_variable)

        ######################################################################################################
        # with splitter
        ######################################################################################################
        train_callback_called = list()

        def train_callback(callback_data_x, callback_data_y, callback_hyper_params):
            assert callback_hyper_params is None
            # ensure the training data has the same len and indexes as the data passed in
            assert len(callback_data_x) == len(train_x)
            assert len(callback_data_y) == len(train_y)
            # ensure the indexes being trained on are the same indexes we manually split above
            assert all(callback_data_x.index.values == train_x.index.values)
            train_callback_called.append('train_called')

        fitter = ModelTrainer(model=LinearRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=train_callback)
        assert len(train_callback_called) == 0
        fitter.train(data=data, target_variable=target_variable, hyper_params=None)
        assert len(train_callback_called) == 1
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        expected_dictionary = {'Mean Absolute Error (MAE)': 8.239183296671666, 'Mean Squared Error (MSE)': 105.83192307350386, 'Root Mean Squared Error (RMSE)': 10.287464365600682, 'RMSE to Standard Deviation of Target': 0.6194779398676552, 'Total Observations': 824}  # noqa
        TestHelper.ensure_values_numeric_dictionary(dictionary_1=expected_dictionary,
                                                    dictionary_2=fitter.training_evaluator.all_quality_metrics)  # noqa

        expected_dictionary = {'Mean Absolute Error (MAE)': 8.381063270249973, 'Mean Squared Error (MSE)': 114.99391155644095, 'Root Mean Squared Error (RMSE)': 10.723521415861534, 'RMSE to Standard Deviation of Target': 0.6287320504285148, 'Total Observations': 206}  # noqa
        TestHelper.ensure_values_numeric_dictionary(dictionary_1=expected_dictionary,
                                                    dictionary_2=fitter.holdout_evaluator.all_quality_metrics)

        ######################################################################################################
        # without splitter
        ######################################################################################################
        train_callback_called = list()

        def train_callback(callback_data_x, callback_data_y, callback_hyper_params):
            assert callback_hyper_params is None
            # now, the length and indexes should match the original dataset, since we are not splitting
            assert len(callback_data_x) == len(data)
            assert len(callback_data_y) == len(data)
            assert all([isclose(x, y) for x, y in zip(callback_data_y, data[target_variable].values)])
            # ensure the indexes being trained on are the same indexes we manually split above
            assert all(callback_data_x.index.values == data.index.values)
            train_callback_called.append('train_called')

        fitter = ModelTrainer(model=LinearRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=None,
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=train_callback)

        assert len(train_callback_called) == 0
        fitter.train(data=data, target_variable=target_variable, hyper_params=None)
        assert len(train_callback_called) == 1
        # if we don't have a splitter, we can still have a training evaluator, but a holdout evaluator (no
        # holdout to evaluate)
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert fitter.holdout_evaluator is None
        # ensure the number of observations in the Training Evaluator matches the number of observations in
        # the entire dataset.
        expected_dictionary = {'Mean Absolute Error (MAE)': 8.264694852478934, 'Mean Squared Error (MSE)': 107.57095170273888, 'Root Mean Squared Error (RMSE)': 10.371641707210044, 'RMSE to Standard Deviation of Target': 0.6211445248863783, 'Total Observations': 1030}  # noqa
        TestHelper.ensure_values_numeric_dictionary(dictionary_1=expected_dictionary,
                                                    dictionary_2=fitter.training_evaluator.all_quality_metrics)  # noqa

    def test_Logistic(self):
        warnings.filterwarnings("ignore")
        # noinspection PyUnusedLocal
        with patch('sys.stdout', new=MockDevice()) as fake_out:  # suppress output of logistic model
            data = TestHelper.get_titanic_data()
            transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                               CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                               ImputationTransformer(),
                               DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

            # test with custom threshold of 0.5
            fitter = ModelTrainer(model=LogisticClassifier(),
                                  model_transformations=transformations,
                                  splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                  evaluator=TwoClassProbabilityEvaluator(
                                     converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)))  # noqa
            fitter.train(data=data, target_variable='Survived', hyper_params=LogisticClassifierHP())
            assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
            assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

            assert fitter.model.hyper_params.params_dict == {'penalty': 'l2', 'regularization_inverse': 1.0, 'solver': 'liblinear'}  # noqa

            con_matrix = fitter.training_evaluator._confusion_matrix
            assert con_matrix.matrix.loc[:, 0].values.tolist() == [386, 85, 471]
            assert con_matrix.matrix.loc[:, 1].values.tolist() == [53, 188, 241]
            assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [439, 273, 712]
            assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
            assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
            assert isclose(round(fitter.training_evaluator.auc_roc, 4), round(0.860346942351498, 4))
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_training_ROC.png',
                                  lambda: fitter.training_evaluator.plot_roc_curve())
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_training_PrecRecal.png',
                                  lambda: fitter.training_evaluator.plot_ppv_tpr_curve())

            con_matrix = fitter.holdout_evaluator._confusion_matrix
            assert con_matrix.matrix.loc[:, 0].values.tolist() == [98, 22, 120]
            assert con_matrix.matrix.loc[:, 1].values.tolist() == [12, 47, 59]
            assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [110, 69, 179]
            assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
            assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
            assert isclose(fitter.holdout_evaluator.auc_roc, 0.8454545454545455)
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_holdout_ROC.png',
                                  lambda: fitter.holdout_evaluator.plot_roc_curve())
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_holdout_PrecRecal.png',
                                  lambda: fitter.holdout_evaluator.plot_ppv_tpr_curve())

            actual_metrics = fitter.training_evaluator.all_quality_metrics
            expected_metrics = {'AUC ROC': 0.860346942351498, 'AUC Precision/Recall': 0.838905616498307, 'Kappa': 0.5807869204973077, 'F1 Score': 0.7315175097276264, 'Two-Class Accuracy': 0.8061797752808989, 'Error Rate': 0.19382022471910113, 'True Positive Rate': 0.6886446886446886, 'True Negative Rate': 0.8792710706150342, 'False Positive Rate': 0.12072892938496584, 'False Negative Rate': 0.31135531135531136, 'Positive Predictive Value': 0.7800829875518672, 'Negative Predictive Value': 0.8195329087048833, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
            assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
            assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

            actual_metrics = fitter.holdout_evaluator.all_quality_metrics
            expected_metrics = {'AUC ROC': 0.8454545454545455, 'AUC Precision/Recall': 0.7920550975664128, 'Kappa': 0.5879485443466486, 'F1 Score': 0.7343750000000001, 'Two-Class Accuracy': 0.8100558659217877, 'Error Rate': 0.18994413407821228, 'True Positive Rate': 0.6811594202898551, 'True Negative Rate': 0.8909090909090909, 'False Positive Rate': 0.10909090909090909, 'False Negative Rate': 0.3188405797101449, 'Positive Predictive Value': 0.7966101694915254, 'Negative Predictive Value': 0.8166666666666667, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
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
        fitter = ModelTrainer(model=LogisticClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=positive_class)))
        fitter.train(data=data, target_variable='Survived', hyper_params=LogisticClassifierHP())
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.hyper_params.params_dict == {'penalty': 'l2', 'regularization_inverse': 1.0, 'solver': 'liblinear'}  # noqa

        con_matrix = fitter.training_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 'died'].values.tolist() == [386, 85, 471]
        assert con_matrix.matrix.loc[:, 'lived'].values.tolist() == [53, 188, 241]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [439, 273, 712]
        assert con_matrix.matrix.index.values.tolist() == ['died', 'lived', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['died', 'lived', 'Total']
        assert isclose(round(fitter.training_evaluator.auc_roc, 4), round(0.860346942351498, 4))

        con_matrix = fitter.holdout_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 'died'].values.tolist() == [98, 22, 120]
        assert con_matrix.matrix.loc[:, 'lived'].values.tolist() == [12, 47, 59]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [110, 69, 179]
        assert con_matrix.matrix.index.values.tolist() == ['died', 'lived', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['died', 'lived', 'Total']
        assert isclose(fitter.holdout_evaluator.auc_roc, 0.8454545454545455)

        actual_metrics = fitter.training_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.860346942351498, 'AUC Precision/Recall': 0.838905616498307, 'Kappa': 0.5807869204973077, 'F1 Score': 0.7315175097276264, 'Two-Class Accuracy': 0.8061797752808989, 'Error Rate': 0.19382022471910113, 'True Positive Rate': 0.6886446886446886, 'True Negative Rate': 0.8792710706150342, 'False Positive Rate': 0.12072892938496584, 'False Negative Rate': 0.31135531135531136, 'Positive Predictive Value': 0.7800829875518672, 'Negative Predictive Value': 0.8195329087048833, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

        actual_metrics = fitter.holdout_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.8454545454545455, 'AUC Precision/Recall': 0.7920550975664128, 'Kappa': 0.5879485443466486, 'F1 Score': 0.7343750000000001, 'Two-Class Accuracy': 0.8100558659217877, 'Error Rate': 0.18994413407821228, 'True Positive Rate': 0.6811594202898551, 'True Negative Rate': 0.8909090909090909, 'False Positive Rate': 0.10909090909090909, 'False Negative Rate': 0.3188405797101449, 'Positive Predictive Value': 0.7966101694915254, 'Negative Predictive Value': 0.8166666666666667, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

    def test_RandomForestHP_classification(self):
        # ensure n_features should be passed in OR all others, but not both
        self.assertRaises(AssertionError, lambda: RandomForestHP(criterion='gini',
                                                                 num_features=100,
                                                                 max_features=100))
        # ensure invalid criterion resampled_stats in assertion error
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
            'oob_score': False}

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
            'oob_score': False}

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
            'oob_score': False}

    def test_RandomForestHP_regression(self):
        # ensure n_features should be passed in OR all others, but not both
        self.assertRaises(AssertionError, lambda: RandomForestHP(criterion='mse',
                                                                 num_features=100,
                                                                 max_features=100))
        # ensure invalid criterion resampled_stats in assertion error
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
            'oob_score': False}

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
            'oob_score': False}

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
            'oob_score': False}

    def test_RandomForestClassifier(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        cache_directory = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_RandomForestMW_classification')  # noqa
        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=RandomForestClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)),
                              persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        assert fitter._persistence_manager._cache_directory == cache_directory
        if os.path.isdir(fitter._persistence_manager._cache_directory):
            shutil.rmtree(fitter._persistence_manager._cache_directory)
        assert not os.path.isdir(fitter._persistence_manager._cache_directory)
        fitter.train(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini'))
        assert os.path.isfile(fitter._persistence_manager._cache_path)

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'n_estimators': 500,
                                                         'criterion': 'gini',
                                                         'max_features': None,
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0,
                                                         'bootstrap': True,
                                                         'oob_score': False}

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.9977638155314692, 'AUC Precision/Recall': 0.9963857279946995, 'Kappa': 0.9642058165548097, 'F1 Score': 0.9777777777777779, 'Two-Class Accuracy': 0.9831460674157303, 'Error Rate': 0.016853932584269662, 'True Positive Rate': 0.967032967032967, 'True Negative Rate': 0.9931662870159453, 'False Positive Rate': 0.00683371298405467, 'False Negative Rate': 0.03296703296703297, 'Positive Predictive Value': 0.9887640449438202, 'Negative Predictive Value': 0.9797752808988764, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.8198945981554676, 'AUC Precision/Recall': 0.7693830218733662, 'Kappa': 0.581636060100167, 'F1 Score': 0.736842105263158, 'Two-Class Accuracy': 0.8044692737430168, 'Error Rate': 0.19553072625698323, 'True Positive Rate': 0.7101449275362319, 'True Negative Rate': 0.8636363636363636, 'False Positive Rate': 0.13636363636363635, 'False Negative Rate': 0.2898550724637681, 'Positive Predictive Value': 0.765625, 'Negative Predictive Value': 0.8260869565217391, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

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
        fitter = ModelTrainer(model=RandomForestClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)),
                              persistence_manager=LocalCacheManager(cache_directory=cache_directory))

        assert fitter._persistence_manager._cache_directory == cache_directory
        shutil.rmtree(fitter._persistence_manager._cache_directory)
        assert not os.path.isdir(fitter._persistence_manager._cache_directory)
        fitter.train(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini',
                                                                                        max_features='auto',
                                                                                        n_estimators=10))
        assert os.path.isfile(fitter._persistence_manager._cache_path)

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'n_estimators': 10,
                                                         'criterion': 'gini',
                                                         'max_features': 'auto',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0,
                                                         'bootstrap': True,
                                                         'oob_score': False}

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.9962785885337139, 'AUC Precision/Recall': 0.9941773735463619, 'Kappa': 0.9491855583543242, 'F1 Score': 0.9683426443202979, 'Two-Class Accuracy': 0.976123595505618, 'Error Rate': 0.023876404494382022, 'True Positive Rate': 0.9523809523809523, 'True Negative Rate': 0.9908883826879271, 'False Positive Rate': 0.009111617312072893, 'False Negative Rate': 0.047619047619047616, 'Positive Predictive Value': 0.9848484848484849, 'Negative Predictive Value': 0.9709821428571429, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.83399209486166, 'AUC Precision/Recall': 0.756480622334744, 'Kappa': 0.5503428610224728, 'F1 Score': 0.7086614173228347, 'Two-Class Accuracy': 0.7932960893854749, 'Error Rate': 0.20670391061452514, 'True Positive Rate': 0.6521739130434783, 'True Negative Rate': 0.8818181818181818, 'False Positive Rate': 0.11818181818181818, 'False Negative Rate': 0.34782608695652173, 'Positive Predictive Value': 0.7758620689655172, 'Negative Predictive Value': 0.8016528925619835, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

    def test_RandomForestClassifier_string_target(self):
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
        fitter = ModelTrainer(model=RandomForestClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=positive_class)))

        fitter.train(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini'))

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'n_estimators': 500,
                                                         'criterion': 'gini',
                                                         'max_features': None,
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0,
                                                         'bootstrap': True,
                                                         'oob_score': False}

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.9977638155314692, 'AUC Precision/Recall': 0.9963857279946995, 'Kappa': 0.9642058165548097, 'F1 Score': 0.9777777777777779, 'Two-Class Accuracy': 0.9831460674157303, 'Error Rate': 0.016853932584269662, 'True Positive Rate': 0.967032967032967, 'True Negative Rate': 0.9931662870159453, 'False Positive Rate': 0.00683371298405467, 'False Negative Rate': 0.03296703296703297, 'Positive Predictive Value': 0.9887640449438202, 'Negative Predictive Value': 0.9797752808988764, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.8198945981554676, 'AUC Precision/Recall': 0.7693830218733662, 'Kappa': 0.581636060100167, 'F1 Score': 0.736842105263158, 'Two-Class Accuracy': 0.8044692737430168, 'Error Rate': 0.19553072625698323, 'True Positive Rate': 0.7101449275362319, 'True Negative Rate': 0.8636363636363636, 'False Positive Rate': 0.13636363636363635, 'False Negative Rate': 0.2898550724637681, 'Positive Predictive Value': 0.765625, 'Negative Predictive Value': 0.8260869565217391, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

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

    def test_RandomForestClassifier_scores(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),
                          SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          SpecificityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)),  # noqa
                          ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))]  # noqa

        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=RandomForestClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              scores=score_list)

        fitter.train(data=data, target_variable='Survived', hyper_params=RandomForestHP(criterion='gini'))

        assert isclose(fitter.training_scores[0].value, 0.9642058165548097)
        assert isclose(fitter.training_scores[1].value, 0.967032967032967)
        assert isclose(fitter.training_scores[2].value, 0.9931662870159453)
        assert isclose(fitter.training_scores[3].value, 0.016853932584269662)

        assert isclose(fitter.holdout_scores[0].value, 0.581636060100167)
        assert isclose(fitter.holdout_scores[1].value, 0.7101449275362319)
        assert isclose(fitter.holdout_scores[2].value, 0.8636363636363636)
        assert isclose(fitter.holdout_scores[3].value, 0.19553072625698323)

    def test_RandomForestRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = [ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=RandomForestRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        fitter.train(data=data, target_variable='strength', hyper_params=RandomForestHP(criterion='MAE', n_estimators=10))  # noqa

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'n_estimators': 10,
                                                         'criterion': 'mae',
                                                         'max_features': None,
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0,
                                                         'bootstrap': True,
                                                         'oob_score': False}

        keys = fitter.training_evaluator.all_quality_metrics.keys()
        expected_values = {'Mean Absolute Error (MAE)': 1.6050794902912628, 'Mean Squared Error (MSE)': 6.946366367415049, 'Root Mean Squared Error (RMSE)': 2.6355960174911193, 'RMSE to Standard Deviation of Target': 0.15718726202422936, 'Total Observations': 824}  # noqa
        assert all([isclose(fitter.training_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

        expected_values = {'Mean Absolute Error (MAE)': 3.7880849514563115, 'Mean Squared Error (MSE)': 29.015298598300976, 'Root Mean Squared Error (RMSE)': 5.3865850590426, 'RMSE to Standard Deviation of Target': 0.3281385595158624, 'Total Observations': 206}  # noqa
        assert all([isclose(fitter.holdout_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

    def test_RandomForestClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=RandomForestClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))

        fitter.train(data=data, target_variable=target_variable, hyper_params=RandomForestHP(criterion='gini',
                                                                                             n_estimators=10,
                                                                                             max_features='auto'))  # noqa

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # noqa
        assert fitter.model.hyper_params.params_dict == {'n_estimators': 10,
                                                         'criterion': 'gini',
                                                         'max_features': 'auto',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0,
                                                         'bootstrap': True,
                                                         'oob_score': False}

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 1.0, 'Accuracy': 1.0, 'Error Rate': 0.0, 'No Information Rate': 0.3392857142857143, 'Total Observations': 112}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.841995841995842, 'Accuracy': 0.8947368421052632, 'Error Rate': 0.10526315789473684, 'No Information Rate': 0.34210526315789475, 'Total Observations': 38}  # noqa

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix['versicolor'].values.tolist() == [0, 12, 3, 15]
        assert con_matrix['virginica'].values.tolist() == [0, 1, 10, 11]
        assert con_matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

    def test_SoftmaxRegression_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=SoftmaxLogisticClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        fitter.train(data=data, target_variable=target_variable, hyper_params=SoftmaxLogisticHP())  # noqa

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length',
                                                   'petal_width']  # noqa
        assert fitter.model.hyper_params.params_dict == {'regularization_inverse': 1.0, 'solver': 'lbfgs'}  # noqa

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 0.959818225304951,
                                                                 'Accuracy': 0.9732142857142857,
                                                                 'Error Rate': 0.0267857142857143,
                                                                 'No Information Rate': 0.3392857142857143,
                                                                 'Total Observations': 112}

        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.920997920997921,
                                                                'Accuracy': 0.9473684210526315,
                                                                'Error Rate': 0.052631578947368474,
                                                                'No Information Rate': 0.34210526315789475,
                                                                'Total Observations': 38}

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix['versicolor'].values.tolist() == [0, 12, 1, 13]
        assert con_matrix['virginica'].values.tolist() == [0, 1, 12, 13]
        assert con_matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

    def test_DummyClassifier_most_frequent(self):
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        fitter = ModelTrainer(model=DummyClassifier(strategy=DummyClassifierStrategy.MOST_FREQUENT),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=TwoClassProbabilityEvaluator(converter=TwoClassThresholdConverter(positive_class=1)))  # noqa
        fitter.train(data=data, target_variable=target_variable)

        data['Survived'].value_counts(normalize=True)

        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.5,
                                                                 'AUC Precision/Recall': 0.38323353293413176,
                                                                 'Kappa': 0.0, 'F1 Score': 0,
                                                                 'Two-Class Accuracy': 0.6167664670658682,
                                                                 'Error Rate': 0.38323353293413176,
                                                                 'True Positive Rate': 0.0,
                                                                 'True Negative Rate': 1.0,
                                                                 'False Positive Rate': 0.0,
                                                                 'False Negative Rate': 1.0,
                                                                 'Positive Predictive Value': 0,
                                                                 'Negative Predictive Value': 0.6167664670658682,  # noqa
                                                                 'Prevalence': 0.38323353293413176,
                                                                 'No Information Rate': 0.6167664670658682,
                                                                 'Total Observations': 668}

        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.5,
                                                                'AUC Precision/Recall': 0.38565022421524664,
                                                                'Kappa': 0.0, 'F1 Score': 0,
                                                                'Two-Class Accuracy': 0.6143497757847534,
                                                                'Error Rate': 0.38565022421524664,
                                                                'True Positive Rate': 0.0,
                                                                'True Negative Rate': 1.0,
                                                                'False Positive Rate': 0.0,
                                                                'False Negative Rate': 1.0,
                                                                'Positive Predictive Value': 0,
                                                                'Negative Predictive Value': 0.6143497757847534,  # noqa
                                                                'Prevalence': 0.38565022421524664,
                                                                'No Information Rate': 0.6143497757847534,
                                                                'Total Observations': 223}

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix[0].values.tolist() == [137, 86, 223]
        assert con_matrix[1].values.tolist() == [0, 0, 0]
        assert con_matrix['Total'].values.tolist() == [137, 86, 223]
        assert con_matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.columns.values.tolist() == [0, 1, 'Total']

    def test_DummyClassifier_stratified(self):
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'

        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        fitter = ModelTrainer(model=DummyClassifier(strategy=DummyClassifierStrategy.STRATIFIED),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=TwoClassProbabilityEvaluator(converter=TwoClassThresholdConverter(positive_class=1)))  # noqa
        fitter.train(data=data, target_variable=target_variable)

        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.5098983616504854,
                                                                 'AUC Precision/Recall': 0.38806072448127005,
                                                                 'Kappa': 0.019767485893891712,
                                                                 'F1 Score': 0.3968871595330739,
                                                                 'Two-Class Accuracy': 0.5359281437125748,
                                                                 'Error Rate': 0.46407185628742514,
                                                                 'True Positive Rate': 0.3984375,
                                                                 'True Negative Rate': 0.6213592233009708,
                                                                 'False Positive Rate': 0.3786407766990291,
                                                                 'False Negative Rate': 0.6015625,
                                                                 'Positive Predictive Value': 0.3953488372093023,  # noqa
                                                                 'Negative Predictive Value': 0.624390243902439,  # noqa
                                                                 'Prevalence': 0.38323353293413176,
                                                                 'No Information Rate': 0.6167664670658682,
                                                                 'Total Observations': 668}

        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.5530470208793073,
                                                                'AUC Precision/Recall': 0.4151358948000934,
                                                                'Kappa': 0.1065552808797204,
                                                                'F1 Score': 0.4470588235294118,
                                                                'Two-Class Accuracy': 0.57847533632287,
                                                                'Error Rate': 0.42152466367713004,
                                                                'True Positive Rate': 0.4418604651162791,
                                                                'True Negative Rate': 0.6642335766423357,
                                                                'False Positive Rate': 0.3357664233576642,
                                                                'False Negative Rate': 0.5581395348837209,
                                                                'Positive Predictive Value': 0.4523809523809524,  # noqa
                                                                'Negative Predictive Value': 0.6546762589928058,  # noqa
                                                                'Prevalence': 0.38565022421524664,
                                                                'No Information Rate': 0.6143497757847534,
                                                                'Total Observations': 223}

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix[0].values.tolist() == [91, 48, 139]
        assert con_matrix[1].values.tolist() == [46, 38, 84]
        assert con_matrix['Total'].values.tolist() == [137, 86, 223]
        assert con_matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.columns.values.tolist() == [0, 1, 'Total']

    def test_SVM_classification(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=SvmLinearClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)))
        fitter.train(data=data, target_variable='Survived', hyper_params=SvmLinearClassifierHP())
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.hyper_params.params_dict == {'penalty': 'l2', 'penalty_c': 1.0, 'loss': 'hinge'}

        con_matrix = fitter.training_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [387, 89, 476]
        assert con_matrix.matrix.loc[:, 1].values.tolist() == [52, 184, 236]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [439, 273, 712]
        assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        assert isclose(fitter.training_evaluator.auc_roc, 0.8621826161689488)

        con_matrix = fitter.holdout_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [99, 27, 126]
        assert con_matrix.matrix.loc[:, 1].values.tolist() == [11, 42, 53]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [110, 69, 179]
        assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        assert isclose(fitter.holdout_evaluator.auc_roc, 0.8363636363636364)

        actual_metrics = fitter.training_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.8621826161689488, 'AUC Precision/Recall': 0.8353822535872308, 'Kappa': 0.5701514009728027, 'F1 Score': 0.7229862475442044, 'Two-Class Accuracy': 0.8019662921348315, 'Error Rate': 0.19803370786516855, 'True Positive Rate': 0.673992673992674, 'True Negative Rate': 0.8815489749430524, 'False Positive Rate': 0.11845102505694761, 'False Negative Rate': 0.326007326007326, 'Positive Predictive Value': 0.7796610169491526, 'Negative Predictive Value': 0.8130252100840336, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

        actual_metrics = fitter.holdout_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.8363636363636364, 'AUC Precision/Recall': 0.7935282490390818, 'Kappa': 0.5316717157807767, 'F1 Score': 0.6885245901639344, 'Two-Class Accuracy': 0.7877094972067039, 'Error Rate': 0.2122905027932961, 'True Positive Rate': 0.6086956521739131, 'True Negative Rate': 0.9, 'False Positive Rate': 0.1, 'False Negative Rate': 0.391304347826087, 'Positive Predictive Value': 0.7924528301886793, 'Negative Predictive Value': 0.7857142857142857, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

    def test_SvmPolynomial_classification(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           CenterScaleTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=SvmPolynomialClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)))
        fitter.train(data=data, target_variable='Survived', hyper_params=SvmPolynomialClassifierHP())
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.hyper_params.params_dict == {'degree': 3, 'coef0': 0.0, 'penalty_c': 1.0}

        con_matrix = fitter.training_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [403, 99, 502]
        assert con_matrix.matrix.loc[:, 1].values.tolist() == [36, 174, 210]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [439, 273, 712]
        assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        assert isclose(fitter.training_evaluator.auc_roc, 0.8623328076631038)

        con_matrix = fitter.holdout_evaluator._confusion_matrix
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [104, 31, 135]
        assert con_matrix.matrix.loc[:, 1].values.tolist() == [6, 38, 44]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [110, 69, 179]
        assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        assert isclose(fitter.holdout_evaluator.auc_roc, 0.8299077733860343)

        actual_metrics = fitter.training_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.8623328076631038, 'AUC Precision/Recall': 0.8282519195039698, 'Kappa': 0.5806941318117573, 'F1 Score': 0.7204968944099378, 'Two-Class Accuracy': 0.8103932584269663, 'Error Rate': 0.1896067415730337, 'True Positive Rate': 0.6373626373626373, 'True Negative Rate': 0.9179954441913439, 'False Positive Rate': 0.08200455580865604, 'False Negative Rate': 0.3626373626373626, 'Positive Predictive Value': 0.8285714285714286, 'Negative Predictive Value': 0.8027888446215139, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

        actual_metrics = fitter.holdout_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.8299077733860343, 'AUC Precision/Recall': 0.7965623013913842, 'Kappa': 0.5321087954786295, 'F1 Score': 0.672566371681416, 'Two-Class Accuracy': 0.7932960893854749, 'Error Rate': 0.20670391061452514, 'True Positive Rate': 0.5507246376811594, 'True Negative Rate': 0.9454545454545454, 'False Positive Rate': 0.05454545454545454, 'False Negative Rate': 0.4492753623188406, 'Positive Predictive Value': 0.8636363636363636, 'Negative Predictive Value': 0.7703703703703704, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

    def test_SvmLinearRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=SvmLinearRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        fitter.train(data=data, target_variable='strength', hyper_params=SvmLinearRegressorHP())

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'epsilon': 0.1, 'penalty_c': 1.0}

        keys = fitter.training_evaluator.all_quality_metrics.keys()
        expected_values = {'Mean Absolute Error (MAE)': 35.78470873786408, 'Mean Squared Error (MSE)': 1561.6856036407662, 'Root Mean Squared Error (RMSE)': 39.518168019795226, 'RMSE to Standard Deviation of Target': 2.3568682719281746, 'Total Observations': 824}  # noqa
        assert all([isclose(fitter.training_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

        expected_values = {'Mean Absolute Error (MAE)': 35.950970873786396, 'Mean Squared Error (MSE)': 1561.9436019417367, 'Root Mean Squared Error (RMSE)': 39.5214321848505, 'RMSE to Standard Deviation of Target': 2.4075561204348044, 'Total Observations': 206}  # noqa
        assert all([isclose(fitter.holdout_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

    def test_SvmPolynomialRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=SvmPolynomialRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        fitter.train(data=data, target_variable='strength', hyper_params=SvmPolynomialRegressorHP())

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'degree': 3, 'epsilon': 0.1, 'penalty_c': 1.0}

        keys = fitter.training_evaluator.all_quality_metrics.keys()
        expected_values = {'Mean Absolute Error (MAE)': 8.89089537374377, 'Mean Squared Error (MSE)': 132.68328212645022, 'Root Mean Squared Error (RMSE)': 11.518822948828157, 'RMSE to Standard Deviation of Target': 0.68698397973439, 'Total Observations': 824}  # noqa
        assert all([isclose(fitter.training_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

        expected_values = {'Mean Absolute Error (MAE)': 9.01999140928503, 'Mean Squared Error (MSE)': 130.26406061226362, 'Root Mean Squared Error (RMSE)': 11.413328200497155, 'RMSE to Standard Deviation of Target': 0.6952740992562293, 'Total Observations': 206}  # noqa
        assert all([isclose(fitter.holdout_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

    def test_CartDecisionTreeRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = None

        fitter = ModelTrainer(model=CartDecisionTreeRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        fitter.train(data=data, target_variable='strength', hyper_params=CartDecisionTreeHP(criterion='mae'))

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'criterion': 'mae',
                                                         'splitter': 'best',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0,
                                                         'max_leaf_nodes': None,
                                                         'max_features': None}

        keys = fitter.training_evaluator.all_quality_metrics.keys()
        expected_values = {'Mean Absolute Error (MAE)': 0.0887135922330097, 'Mean Squared Error (MSE)': 1.608594963592233, 'Root Mean Squared Error (RMSE)': 1.2683039712908861, 'RMSE to Standard Deviation of Target': 0.07564180069275088, 'Total Observations': 824}  # noqa
        assert all([isclose(fitter.training_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

        expected_values = {'Mean Absolute Error (MAE)': 4.878082524271845, 'Mean Squared Error (MSE)': 65.98628021844661, 'Root Mean Squared Error (RMSE)': 8.123193966565529, 'RMSE to Standard Deviation of Target': 0.4948465748966606, 'Total Observations': 206}  # noqa
        assert all([isclose(fitter.holdout_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

    def test_CartDecisionTreeClassifier(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        fitter = ModelTrainer(model=CartDecisionTreeClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)))
        fitter.train(data=data, target_variable='Survived', hyper_params=CartDecisionTreeHP(criterion='gini'))
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'criterion': 'gini',
                                                         'splitter': 'best',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0,
                                                         'max_leaf_nodes': None,
                                                         'max_features': None}

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.9992991063606098, 'AUC Precision/Recall': 0.9984018681159533, 'Kappa': 0.9641059680549836, 'F1 Score': 0.9776119402985075, 'Two-Class Accuracy': 0.9831460674157303, 'Error Rate': 0.016853932584269662, 'True Positive Rate': 0.9597069597069597, 'True Negative Rate': 0.9977220956719818, 'False Positive Rate': 0.002277904328018223, 'False Negative Rate': 0.040293040293040296, 'Positive Predictive Value': 0.9961977186311787, 'Negative Predictive Value': 0.9755011135857461, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.7711462450592885, 'AUC Precision/Recall': 0.6435502175777592, 'Kappa': 0.5601381417281001, 'F1 Score': 0.725925925925926, 'Two-Class Accuracy': 0.7932960893854749, 'Error Rate': 0.20670391061452514, 'True Positive Rate': 0.7101449275362319, 'True Negative Rate': 0.8454545454545455, 'False Positive Rate': 0.15454545454545454, 'False Negative Rate': 0.2898550724637681, 'Positive Predictive Value': 0.7424242424242424, 'Negative Predictive Value': 0.8230088495575221, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

    def test_CartDecisionTreeClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=CartDecisionTreeClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        fitter.train(data=data, target_variable=target_variable, hyper_params=CartDecisionTreeHP(criterion='gini'))  # noqa

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # noqa
        assert fitter.model.hyper_params.params_dict == {'criterion': 'gini',
                                                         'splitter': 'best',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0,
                                                         'max_leaf_nodes': None,
                                                         'max_features': None}

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 1.0, 'Accuracy': 1.0, 'Error Rate': 0.0, 'No Information Rate': 0.3392857142857143, 'Total Observations': 112}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.841995841995842, 'Accuracy': 0.8947368421052632, 'Error Rate': 0.10526315789473684, 'No Information Rate': 0.34210526315789475, 'Total Observations': 38}  # noqa

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix['versicolor'].values.tolist() == [0, 12, 3, 15]
        assert con_matrix['virginica'].values.tolist() == [0, 1, 10, 11]
        assert con_matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

    def test_AdaBoostRegressor(self):
        data = TestHelper.get_cement_data()

        fitter = ModelTrainer(model=AdaBoostRegressor(),
                              model_transformations=None,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        fitter.train(data=data, target_variable='strength', hyper_params=AdaBoostRegressorHP())

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'max_depth': None,
                                                         'n_estimators': 50,
                                                         'learning_rate': 1.0,
                                                         'loss': 'linear'}

        keys = fitter.training_evaluator.all_quality_metrics.keys()
        expected_values = {'Mean Absolute Error (MAE)': 1.5431027895916443, 'Mean Squared Error (MSE)': 7.862081902399274, 'Root Mean Squared Error (RMSE)': 2.8039404241886587, 'RMSE to Standard Deviation of Target': 0.16722734259434227, 'Total Observations': 824}  # noqa
        assert all([isclose(fitter.training_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

        expected_values = {'Mean Absolute Error (MAE)': 3.9066797652274916, 'Mean Squared Error (MSE)': 30.344412064467065, 'Root Mean Squared Error (RMSE)': 5.508576228433902, 'RMSE to Standard Deviation of Target': 0.33556998520745485, 'Total Observations': 206}  # noqa
        assert all([isclose(fitter.holdout_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

    def test_AdaBoostClassifier(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        fitter = ModelTrainer(model=AdaBoostClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)))
        fitter.train(data=data, target_variable='Survived', hyper_params=AdaBoostClassifierHP())
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'max_depth': None,
                                                         'n_estimators': 50,
                                                         'learning_rate': 1.0,
                                                         'algorithm': 'SAMME.R'}

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.9992991063606097, 'AUC Precision/Recall': 0.9984018681159533, 'Kappa': 0.9641059680549836, 'F1 Score': 0.9776119402985075, 'Two-Class Accuracy': 0.9831460674157303, 'Error Rate': 0.016853932584269662, 'True Positive Rate': 0.9597069597069597, 'True Negative Rate': 0.9977220956719818, 'False Positive Rate': 0.002277904328018223, 'False Negative Rate': 0.040293040293040296, 'Positive Predictive Value': 0.9961977186311787, 'Negative Predictive Value': 0.9755011135857461, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.797957839262187, 'AUC Precision/Recall': 0.7339255444753214, 'Kappa': 0.5770035784214436, 'F1 Score': 0.7286821705426356, 'Two-Class Accuracy': 0.8044692737430168, 'Error Rate': 0.19553072625698323, 'True Positive Rate': 0.6811594202898551, 'True Negative Rate': 0.8818181818181818, 'False Positive Rate': 0.11818181818181818, 'False Negative Rate': 0.3188405797101449, 'Positive Predictive Value': 0.7833333333333333, 'Negative Predictive Value': 0.8151260504201681, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

    def test_AdaBoostClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=AdaBoostClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        fitter.train(data=data, target_variable=target_variable, hyper_params=AdaBoostClassifierHP())  # noqa

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # noqa
        assert fitter.model.hyper_params.params_dict == {'max_depth': None,
                                                         'n_estimators': 50,
                                                         'learning_rate': 1.0,
                                                         'algorithm': 'SAMME.R'}

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 1.0, 'Accuracy': 1.0, 'Error Rate': 0.0, 'No Information Rate': 0.3392857142857143, 'Total Observations': 112}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.841995841995842, 'Accuracy': 0.8947368421052632, 'Error Rate': 0.10526315789473684, 'No Information Rate': 0.34210526315789475, 'Total Observations': 38}  # noqa

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix['versicolor'].values.tolist() == [0, 12, 3, 15]
        assert con_matrix['virginica'].values.tolist() == [0, 1, 10, 11]
        assert con_matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

    def test_GradientBoostingRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = None

        fitter = ModelTrainer(model=GradientBoostingRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        fitter.train(data=data, target_variable='strength', hyper_params=GradientBoostingRegressorHP())

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa
        assert fitter.model.hyper_params.params_dict == {'loss': 'ls',
                                                         'learning_rate': 0.1,
                                                         'n_estimators': 100,
                                                         'max_depth': 3,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': None}

        keys = fitter.training_evaluator.all_quality_metrics.keys()
        expected_values = {'Mean Absolute Error (MAE)': 2.921039389435212, 'Mean Squared Error (MSE)': 14.909877151863128, 'Root Mean Squared Error (RMSE)': 3.861331007808464, 'RMSE to Standard Deviation of Target': 0.23029024359523864, 'Total Observations': 824}  # noqa
        assert all([isclose(fitter.training_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

        expected_values = {'Mean Absolute Error (MAE)': 4.004637648118118, 'Mean Squared Error (MSE)': 26.996454670801214, 'Root Mean Squared Error (RMSE)': 5.195811262045727, 'RMSE to Standard Deviation of Target': 0.3165170519644618, 'Total Observations': 206}  # noqa
        assert all([isclose(fitter.holdout_evaluator.all_quality_metrics[x], expected_values[x]) for x in keys])  # noqa

    def test_GradientBoostingClassifier(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        fitter = ModelTrainer(model=GradientBoostingClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)))
        fitter.train(data=data, target_variable='Survived', hyper_params=GradientBoostingClassifierHP())
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'loss': 'deviance',
                                                         'learning_rate': 0.1,
                                                         'n_estimators': 100,
                                                         'max_depth': 3,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': None}

        assert fitter.training_evaluator.all_quality_metrics == {'AUC ROC': 0.9551594950228207, 'AUC Precision/Recall': 0.9422935585546288, 'Kappa': 0.7902449021416128, 'F1 Score': 0.8654970760233918, 'Two-Class Accuracy': 0.9030898876404494, 'Error Rate': 0.09691011235955056, 'True Positive Rate': 0.8131868131868132, 'True Negative Rate': 0.958997722095672, 'False Positive Rate': 0.04100227790432802, 'False Negative Rate': 0.18681318681318682, 'Positive Predictive Value': 0.925, 'Negative Predictive Value': 0.8919491525423728, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'AUC ROC': 0.8112648221343873, 'AUC Precision/Recall': 0.7673499978743319, 'Kappa': 0.5503428610224728, 'F1 Score': 0.7086614173228347, 'Two-Class Accuracy': 0.7932960893854749, 'Error Rate': 0.20670391061452514, 'True Positive Rate': 0.6521739130434783, 'True Negative Rate': 0.8818181818181818, 'False Positive Rate': 0.11818181818181818, 'False Negative Rate': 0.34782608695652173, 'Positive Predictive Value': 0.7758620689655172, 'Negative Predictive Value': 0.8016528925619835, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa

    def test_GradientBoostingClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=GradientBoostingClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        fitter.train(data=data, target_variable=target_variable, hyper_params=GradientBoostingClassifierHP())  # noqa

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # noqa
        assert fitter.model.hyper_params.params_dict == {'loss': 'deviance',
                                                         'learning_rate': 0.1,
                                                         'n_estimators': 100,
                                                         'max_depth': 3,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': None}

        assert fitter.training_evaluator.all_quality_metrics == {'Kappa': 1.0, 'Accuracy': 1.0, 'Error Rate': 0.0, 'No Information Rate': 0.3392857142857143, 'Total Observations': 112}  # noqa
        assert fitter.holdout_evaluator.all_quality_metrics == {'Kappa': 0.9604989604989606, 'Accuracy': 0.9736842105263158, 'Error Rate': 0.02631578947368418, 'No Information Rate': 0.34210526315789475, 'Total Observations': 38}  # noqa

        con_matrix = fitter.holdout_evaluator.matrix
        assert con_matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix['versicolor'].values.tolist() == [0, 13, 1, 14]
        assert con_matrix['virginica'].values.tolist() == [0, 0, 12, 12]
        assert con_matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

    def test_ModelAggregator(self):
        data = TestHelper.get_titanic_data()
        data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Age', 'Embarked'], inplace=True)
        data.Sex = [1 if x == 'male' else 0 for x in data.Sex]
        # data.Survived = [ if x == 'male' else 0 for x in data.Sex]
        target_variable = 'Survived'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_class(data, target_variable)

        ######################################################################################################
        # Build Classifiers that will be using by the ModelAggregator (must be pre-trained)
        ######################################################################################################
        model_random_forest = RandomForestClassifier()
        model_random_forest.train(data_x=train_x, data_y=train_y, hyper_params=RandomForestHP())

        model_decision_tree = CartDecisionTreeClassifier()
        model_decision_tree.train(data_x=train_x, data_y=train_y, hyper_params=CartDecisionTreeHP())

        model_adaboost = AdaBoostClassifier()
        model_adaboost.train(data_x=train_x, data_y=train_y, hyper_params=AdaBoostClassifierHP())

        predictions_random_forest = model_random_forest.predict(data_x=holdout_x)
        predictions_decision_tree = model_decision_tree.predict(data_x=holdout_x)
        predictions_adaboost = model_adaboost.predict(data_x=holdout_x)
        # model_predictions = [predictions_random_forest, predictions_decision_tree, predictions_adaboost]
        died_averages = np.mean([predictions_random_forest[0].values, predictions_decision_tree[0].values, predictions_adaboost[0].values], axis=0)  # noqa
        survived_averages = np.mean([predictions_random_forest[1].values, predictions_decision_tree[1].values, predictions_adaboost[1].values], axis=0)  # noqa

        assert isclose(roc_auc_score(y_true=holdout_y, y_score=predictions_random_forest[1]), 0.8230566534914362)  # noqa
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_decision_tree.predict(data_x=holdout_x)[1]), 0.772463768115942)  # noqa
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_adaboost.predict(data_x=holdout_x)[1]), 0.77832674571805)  # noqa

        ######################################################################################################
        # VotingStrategy.SOFT
        ######################################################################################################
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=SoftVotingAggregationStrategy())
        # `train()` does nothing, but make sure it doesn't explode in case it is used in a process that
        # automatically calls `train()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_soft.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(voting_predictions, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_predictions = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_predictions,
                                                      data_frame2=voting_predictions)

        # make sure we are getting the correct averages back
        assert all([isclose(x, y) for x, y in zip(voting_predictions[0], died_averages)])
        assert all([isclose(x, y) for x, y in zip(voting_predictions[1], survived_averages)])
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_aggregator.predict(data_x=holdout_x)[1]), 0.8073781291172596)  # noqa

        ######################################################################################################
        # VotingStrategy.HARD
        ######################################################################################################
        # for HARD voting, need to pass converters as well
        converter = TwoClassThresholdConverter(positive_class=1)
        # noinspection PyUnusedLocal
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        # noinspection PyUnusedLocal
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=HardVotingAggregationStrategy(converters=[copy.deepcopy(converter) for x in range(0, 3)]))  # noqa
        # `train()` does nothing, but make sure it doesn't explode in case it is used in a process that
        # automatically calls `train()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_hard.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(voting_predictions, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_predictions = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_predictions,
                                                      data_frame2=voting_predictions)

        # predictions should all sum to 1
        assert all([sum(voting_predictions.loc[x]) == 1.0 for x in voting_predictions.index.values])

        # [1 if x > 0.5 else 0 for x in predictions_random_forest[1]]
        # [1 if x > 0.5 else 0 for x in model_aggregator.predict(data_x=holdout_x)[1]]
        # list(holdout_y)
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_aggregator.predict(data_x=holdout_x)[1]), 0.7853096179183136)  # noqa

    def test_ModelAggregator_multi_class(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_class(data, target_variable)

        ######################################################################################################
        # Build Classifiers that will be using by the ModelAggregator (must be pre-trained)
        ######################################################################################################
        model_random_forest = RandomForestClassifier()
        model_random_forest.train(data_x=train_x, data_y=train_y, hyper_params=RandomForestHP())

        model_decision_tree = CartDecisionTreeClassifier()
        model_decision_tree.train(data_x=train_x, data_y=train_y, hyper_params=CartDecisionTreeHP())

        model_adaboost = AdaBoostClassifier()
        model_adaboost.train(data_x=train_x, data_y=train_y, hyper_params=AdaBoostClassifierHP())

        predictions_random_forest = model_random_forest.predict(data_x=holdout_x)
        predictions_decision_tree = model_decision_tree.predict(data_x=holdout_x)
        predictions_adaboost = model_adaboost.predict(data_x=holdout_x)
        # model_predictions = [predictions_random_forest, predictions_decision_tree, predictions_adaboost]

        setosa_averages = np.mean([predictions_random_forest['setosa'].values, predictions_decision_tree['setosa'].values, predictions_adaboost['setosa'].values], axis=0)  # noqa
        versicolor_averages = np.mean([predictions_random_forest['versicolor'].values, predictions_decision_tree['versicolor'].values, predictions_adaboost['versicolor'].values], axis=0)  # noqa
        virginica_averages = np.mean([predictions_random_forest['virginica'].values, predictions_decision_tree['virginica'].values, predictions_adaboost['virginica'].values], axis=0)  # noqa

        evaluator = MultiClassEvaluator(converter=HighestValueConverter())
        evaluator.evaluate(actual_values=holdout_y, predicted_values=predictions_random_forest)
        assert evaluator.all_quality_metrics == {'Kappa': 0.95, 'Accuracy': 0.9666666666666667, 'Error Rate': 0.033333333333333326, 'No Information Rate': 0.3333333333333333, 'Total Observations': 30}  # noqa

        evaluator.evaluate(actual_values=holdout_y, predicted_values=predictions_decision_tree)
        assert evaluator.all_quality_metrics == {'Kappa': 0.9, 'Accuracy': 0.9333333333333333, 'Error Rate': 0.06666666666666665, 'No Information Rate': 0.3333333333333333, 'Total Observations': 30}  # noqa

        evaluator.evaluate(actual_values=holdout_y, predicted_values=predictions_adaboost)
        assert evaluator.all_quality_metrics == {'Kappa': 0.9, 'Accuracy': 0.9333333333333333, 'Error Rate': 0.06666666666666665, 'No Information Rate': 0.3333333333333333, 'Total Observations': 30}  # noqa

        ######################################################################################################
        # VotingStrategy.SOFT
        ######################################################################################################
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=SoftVotingAggregationStrategy())
        # `train()` does nothing, but make sure it doesn't explode in case it is used in a process that
        # automatically calls `train()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        # make sure we are getting the correct averages back
        assert all([isclose(x, y) for x, y in zip(voting_predictions['setosa'], setosa_averages)])
        assert all([isclose(x, y) for x, y in zip(voting_predictions['versicolor'], versicolor_averages)])
        assert all([isclose(x, y) for x, y in zip(voting_predictions['virginica'], virginica_averages)])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_multi_class_soft.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(voting_predictions, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_predictions = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_predictions,
                                                      data_frame2=voting_predictions)

        evaluator.evaluate(actual_values=holdout_y, predicted_values=voting_predictions)
        assert evaluator.all_quality_metrics == {'Kappa': 0.9, 'Accuracy': 0.9333333333333333, 'Error Rate': 0.06666666666666665, 'No Information Rate': 0.3333333333333333, 'Total Observations': 30}  # noqa

        ######################################################################################################
        # VotingStrategy.HARD
        ######################################################################################################
        # for HARD voting, need to pass converters as well
        converter = HighestValueConverter()
        # noinspection PyUnusedLocal
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        # noinspection PyUnusedLocal
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=HardVotingAggregationStrategy(converters=[copy.deepcopy(converter) for x in range(0, 3)]))  # noqa
        # `train()` does nothing, but make sure it doesn't explode in case it is used in a process that
        # automatically calls `train()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        # predictions should all sum to 1
        assert all([sum(voting_predictions.loc[x]) == 1.0 for x in voting_predictions.index.values])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_multi_class_hard.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(voting_predictions, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_predictions = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_predictions,
                                                      data_frame2=voting_predictions)

        # [1 if x > 0.5 else 0 for x in predictions_random_forest[1]]
        # [1 if x > 0.5 else 0 for x in model_aggregator.predict(data_x=holdout_x)[1]]
        # list(holdout_y)

        evaluator.evaluate(actual_values=holdout_y, predicted_values=voting_predictions)
        assert evaluator.all_quality_metrics == {'Kappa': 0.9, 'Accuracy': 0.9333333333333333, 'Error Rate': 0.06666666666666665, 'No Information Rate': 0.3333333333333333, 'Total Observations': 30}  # noqa

    def test_ModelAggregator_regression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_regression(data, target_variable)  # noqa

        model_linear_regression = LinearRegressor()
        model_linear_regression.train(data_x=train_x, data_y=train_y)
        predictions_linear_regression = model_linear_regression.predict(data_x=holdout_x)

        model_cart = CartDecisionTreeRegressor()
        model_cart.train(data_x=train_x, data_y=train_y, hyper_params=CartDecisionTreeHP(criterion='mse'))
        predictions_cart = model_cart.predict(data_x=holdout_x)

        model_adaboost = AdaBoostRegressor()
        model_adaboost.train(data_x=train_x, data_y=train_y, hyper_params=AdaBoostRegressorHP())
        predictions_adaboost = model_adaboost.predict(data_x=holdout_x)

        expected_aggregation = (predictions_linear_regression + predictions_cart + predictions_adaboost) / 3

        model_infos = [ModelInfo(model=LinearRegressor(), hyper_params=None),
                       ModelInfo(model=CartDecisionTreeRegressor(), hyper_params=CartDecisionTreeHP(criterion='mse')),  # noqa
                       ModelInfo(model=AdaBoostRegressor(), hyper_params=AdaBoostRegressorHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=MeanAggregationStrategy())
        model_aggregator.train(data_x=train_x, data_y=train_y)
        predictions_aggregation = model_aggregator.predict(data_x=holdout_x)

        assert all([isclose(x, y) for x, y in zip(expected_aggregation, predictions_aggregation)])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_regression.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(expected_aggregation, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_predictions = pickle.load(saved_object)
            assert all([isclose(x, y) for x, y in zip(expected_predictions, predictions_aggregation)])

    def test_ModelAggregator_regression_transformations(self):
        """
        Make sure transformations work as expected for each
        """
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_regression(data,
                                                                                           target_variable)
        model_infos = [ModelInfo(model=LinearRegressor(),
                                 hyper_params=None,
                                 transformations=[PolynomialFeaturesTransformer(degrees=3),
                                                  DummyEncodeTransformer(CategoricalEncoding.DUMMY)]),
                       ModelInfo(model=CartDecisionTreeRegressor(),
                                 hyper_params=CartDecisionTreeHP(criterion='mse'),
                                 transformations=[CenterScaleTransformer(),
                                                  DummyEncodeTransformer(CategoricalEncoding.DUMMY)]),
                       ModelInfo(model=AdaBoostRegressor(), hyper_params=AdaBoostRegressorHP(),
                                 transformations=[DummyEncodeTransformer(CategoricalEncoding.DUMMY)])]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=MeanAggregationStrategy())
        model_aggregator.train(data_x=train_x, data_y=train_y)

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_LinearRegressor.pkl'),  # noqa
                                                     expected_dataframe=model_aggregator._base_models[0].model.data_x_trained_head)  # noqa

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_CartDecisionTreeRegressor.pkl'),  # noqa
                                                     expected_dataframe=model_aggregator._base_models[1].model.data_x_trained_head)  # noqa

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_AdaBoostRegressor.pkl'),  # noqa
                                                     expected_dataframe=model_aggregator._base_models[2].model.data_x_trained_head)  # noqa

        predictions_aggregation = model_aggregator.predict(data_x=holdout_x)
        evaluator = RegressionEvaluator()
        evaluator.evaluate(actual_values=holdout_y, predicted_values=predictions_aggregation)
        TestHelper.ensure_values_numeric_dictionary(dictionary_1=evaluator.all_quality_metrics,
                                                    dictionary_2={'Mean Absolute Error (MAE)': 2929.902780386074, 'Mean Squared Error (MSE)': 26119870.737061337, 'Root Mean Squared Error (RMSE)': 5110.7602895324035, 'RMSE to Standard Deviation of Target': 0.4126739290232337, 'Total Observations': 268})  # noqa

    def test_ModelAggregator_regression_median_transformations(self):
        """
        Make sure transformations work as expected for each
        """
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_regression(data,
                                                                                           target_variable)
        model_infos = [ModelInfo(model=LinearRegressor(),
                                 hyper_params=None,
                                 transformations=[PolynomialFeaturesTransformer(degrees=3),
                                                  DummyEncodeTransformer(CategoricalEncoding.DUMMY)]),
                       ModelInfo(model=CartDecisionTreeRegressor(),
                                 hyper_params=CartDecisionTreeHP(criterion='mse'),
                                 transformations=[CenterScaleTransformer(),
                                                  DummyEncodeTransformer(CategoricalEncoding.DUMMY)]),
                       ModelInfo(model=AdaBoostRegressor(), hyper_params=AdaBoostRegressorHP(),
                                 transformations=[DummyEncodeTransformer(CategoricalEncoding.DUMMY)])]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=MedianAggregationStrategy())
        model_aggregator.train(data_x=train_x, data_y=train_y)

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_LinearRegressor.pkl'),  # noqa
                                                     expected_dataframe=model_aggregator._base_models[0].model.data_x_trained_head)  # noqa

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_CartDecisionTreeRegressor.pkl'),  # noqa
                                                     expected_dataframe=model_aggregator._base_models[1].model.data_x_trained_head)  # noqa

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelAggregator_AdaBoostRegressor.pkl'),  # noqa
                                                     expected_dataframe=model_aggregator._base_models[2].model.data_x_trained_head)  # noqa

        predictions_aggregation = model_aggregator.predict(data_x=holdout_x)
        evaluator = RegressionEvaluator()
        evaluator.evaluate(actual_values=holdout_y, predicted_values=predictions_aggregation)
        TestHelper.ensure_values_numeric_dictionary(dictionary_1=evaluator.all_quality_metrics,
                                                    dictionary_2={'Mean Absolute Error (MAE)': 2219.7179684340335, 'Mean Squared Error (MSE)': 24703837.75169208, 'Root Mean Squared Error (RMSE)': 4970.295539672876, 'RMSE to Standard Deviation of Target': 0.4013319491748498, 'Total Observations': 268})  # noqa

    def helper_test_ModelStacker_Classification(self, data, positive_class, cache_directory=None):
        """
        NOTE: WHEN USING cache_directory THIS FUNCTION DOESN'T CHECK WHETHER OR NOT THE CACHED FILES EXIST.
            SO YOU CAN TEST AS IF RUNNING THE SEARCHER FOR "FIRST TIME", OR "SUBSEQUENT RUNS" (i.e. cache
            files already exist)
        """
        target_variable = 'Survived'
        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class)),  # noqa
                      SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class))]  # noqa

        cart_base_model = ModelInfo(description='cart',
                                    model=CartDecisionTreeClassifier(),
                                    transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                    hyper_params=CartDecisionTreeHP(),
                                    converter=ExtractPredictionsColumnConverter(column=positive_class))
        rf_base_model = ModelInfo(description='random_forest',
                                  model=RandomForestClassifier(),
                                  transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                  hyper_params=RandomForestHP(),
                                  converter=ExtractPredictionsColumnConverter(column=positive_class))
        base_models = [cart_base_model, rf_base_model]

        # Test model stacker with duplicate model names; should get assertion error
        self.assertRaises(AssertionError,
                          lambda: ModelStacker(base_models=base_models + [cart_base_model],
                                               scores=score_list, stacking_model=LogisticClassifier()))

        # Use same splitter information to get the training/holdout data
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)
        expected_train_y = data.iloc[training_indexes].Survived
        expected_train_x = data.iloc[training_indexes].drop(columns=target_variable)
        expected_test_x = data.iloc[test_indexes].drop(columns=target_variable)

        file_train_callback = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_train_callback_train_meta.pkl'))  # noqa
        file_test_callback_traindata = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_test_callback_test_meta_traindata.pkl'))  # noqa
        file_test_callback_holdoutdata = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_test_callback_test_meta_holdoutdata.pkl'))  # noqa

        # these variables help us verify the callbacks were called
        train_callback_called = list()
        predict_callback_called = list()

        def train_callback(train_meta_x, train_meta_y, hyper_params):
            # the `train_meta_x` that we built in `train()` should have the same indexes as `train_meta_x
            assert all(train_meta_x.index.values == expected_train_x.index.values)
            # cached dataframe in `file_train_callback` are the predictions from the base model we expect
            TestHelper.ensure_all_values_equal_from_file(file=file_train_callback,
                                                         expected_dataframe=train_meta_x)
            assert all(train_meta_y.index.values == expected_train_x.index.values)
            # check that the y values in `train_meta_y` which will be trained on match the y's from the split
            assert all(train_meta_y == expected_train_y)
            assert isinstance(hyper_params, LogisticClassifierHP)

            # add value to the list so we know this callback was called and completed
            train_callback_called.append('train_called')

        # predict will be called twice, first for evaluating the training data; next for the holdout
        def predict_callback(test_meta):
            if len(predict_callback_called) == 0:  # first time called i.e. training evaluation
                # the `test_meta` that we built in `predict()` should have the same indexes as `train_meta_x`
                # when evaluating the training set
                assert all(test_meta.index.values == expected_train_x.index.values)
                # cached dataframe in `file_test_callback_traindata` are the predictions from the base
                # models (refitted on the entire training set) on the dataset that was passed into `predict()`
                # in this case the original training data, which will then be predicted on the final stacking
                # model
                TestHelper.ensure_all_values_equal_from_file(file=file_test_callback_traindata,
                                                             expected_dataframe=test_meta)
                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_train')

            elif len(predict_callback_called) == 1:  # second time called i.e. holdout evaluation
                # the `test_meta` that we built in `predict()` should have the same indexes as
                # `expected_test_x` when evaluating the holdout set
                assert all(test_meta.index.values == expected_test_x.index.values)
                # cached dataframe in `file_test_callback_holdoutdata` are the predictions from the base
                # models (refitted on the entire training set) on the dataset that was passed into `predict()`
                # in this case the holdout data, which will then be predicted on the final stacking model
                TestHelper.ensure_all_values_equal_from_file(file=file_test_callback_holdoutdata,
                                                             expected_dataframe=test_meta)

                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_holdout')
            else:
                raise ValueError()

        model_stacker = ModelStacker(base_models=base_models,
                                     scores=score_list,
                                     stacking_model=LogisticClassifier(),
                                     stacking_transformations=None,
                                     train_callback=train_callback,
                                     predict_callback=predict_callback)

        assert len(train_callback_called) == 0
        assert len(predict_callback_called) == 0

        if cache_directory:
            # NOTE: THIS DOESN'T CHECK WHETHER OR NOT THE CACHED FILES EXIST. CAN TEST FOR FIRST TIME, OR NOT
            fitter = ModelTrainer(model=model_stacker,
                                  # transformations for all models, not just stackers
                                  model_transformations=[RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),  # noqa
                                                        CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),  # noqa
                                                        ImputationTransformer()],
                                  splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                  evaluator=TwoClassProbabilityEvaluator(
                                     converter=TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=positive_class)),
                                  persistence_manager=LocalCacheManager(cache_directory=cache_directory))
            assert fitter._persistence_manager._cache_directory == cache_directory

            time_start = time.time()
            fitter.train(data=data, target_variable=target_variable, hyper_params=LogisticClassifierHP())
            time_stop = time.time()
            fit_time = time_stop - time_start

            assert os.path.isdir(fitter._persistence_manager._cache_directory)
            expected_stacker_cached_file = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification/ModelStacker_LogisticClassifier_penalty_l2_regularization_inverse_1.0_solver_liblinear.pkl')  # noqa
            expected_base_cart_cached_file = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification/base_cart.pkl')  # noqa
            expected_base_rf_cached_file = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification/base_random_forest.pkl')  # noqa
            # ensure the cache path of the stacker is the final stacked model
            assert fitter._persistence_manager._cache_path == expected_stacker_cached_file
            assert os.path.isfile(fitter._persistence_manager._cache_path)
            # each base model should have the following file,
            # corresponding to the final trained model for each base model
            assert os.path.isfile(expected_base_cart_cached_file)
            assert os.path.isfile(expected_base_rf_cached_file)
            # each base model should have a corresponding directory
            assert os.path.isdir(os.path.join(fitter._persistence_manager._cache_directory, 'resample_' + base_models[0].description))  # noqa
            assert os.path.isdir(os.path.join(fitter._persistence_manager._cache_directory, 'resample_' + base_models[1].description))  # noqa
            # check the directory contents of the base models
            cart_pkl_file_name = 'CartDecisionTreeClassifier_criteriongini_splitterbest_max_depthNone_min_samples_split2_min_samples_leaf1_min_weight_fraction_leaf0.0_max_leaf_nodesNone_max_featuresNone.pkl'  # noqa
            rf_pkl_file_name = 'RandomForestClassifier_n_estimators500_criteriongini_max_featuresNone_max_depthNone_min_samples_split2_min_samples_leaf1_min_weight_fraction_leaf0.0_max_leaf_nodesNone_min_impurity_decrease0_bootstrapTrue_oob_scoreFalse.pkl'  # noqa
            # there should only be 1 repeat and 5 folds
            expected_prefixes = ['repeat0_fold' + str(x) for x in range(5)]
            # check that all the expected files exist for the given directory and model/hyper-params
            assert all([os.path.isfile(os.path.join(cache_directory, 'resample_cart', "{0}_{1}".format(x, cart_pkl_file_name))) for x in expected_prefixes])  # noqa
            assert all([os.path.isfile(os.path.join(cache_directory, 'resample_random_forest', "{0}_{1}".format(x, rf_pkl_file_name))) for x in expected_prefixes])  # noqa

        else:
            fitter = ModelTrainer(model=model_stacker,
                                  # transformations for all models, not just stackers
                                  model_transformations=[RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),  # noqa
                                                        CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),  # noqa
                                                        ImputationTransformer()],
                                  splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                  evaluator=TwoClassProbabilityEvaluator(
                                     converter=TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=positive_class)))
            time_start = time.time()
            fitter.train(data=data, target_variable=target_variable, hyper_params=LogisticClassifierHP())
            time_stop = time.time()
            fit_time = time_stop - time_start

        # verify our callback is called. If it wasn't, we would never know and the assertions wouldn't run.
        assert train_callback_called == ['train_called']
        # `predict_callback` should be called TWICE (once for training eval & once for holdout eval)
        assert predict_callback_called == ['predict_called_train', 'predict_called_holdout']

        actual_metrics = fitter.training_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.9976052800654167, 'AUC Precision/Recall': 0.9961793020728291, 'Kappa': 0.9641559618401953, 'F1 Score': 0.9776951672862454, 'Two-Class Accuracy': 0.9831460674157303, 'Error Rate': 0.016853932584269662, 'True Positive Rate': 0.9633699633699634, 'True Negative Rate': 0.9954441913439636, 'False Positive Rate': 0.004555808656036446, 'False Negative Rate': 0.03663003663003663, 'Positive Predictive Value': 0.9924528301886792, 'Negative Predictive Value': 0.9776286353467561, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])
        # holdout AUC/ROC for CART was: 0.7711462450592885; for Random Forest was 0.8198945981554676;
        # slight increase to 0.8203557312252964; (default hyper-params for all models)
        actual_metrics = fitter.holdout_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.8203557312252964, 'AUC Precision/Recall': 0.7712259990480193, 'Kappa': 0.5793325723494259, 'F1 Score': 0.732824427480916, 'Two-Class Accuracy': 0.8044692737430168, 'Error Rate': 0.19553072625698323, 'True Positive Rate': 0.6956521739130435, 'True Negative Rate': 0.8727272727272727, 'False Positive Rate': 0.12727272727272726, 'False Negative Rate': 0.30434782608695654, 'Positive Predictive Value': 0.7741935483870968, 'Negative Predictive Value': 0.8205128205128205, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

        # test resample data
        file_test_model_stacker_resample_data_cart = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_cart.pkl'))  # noqa
        file_test_model_stacker_resample_data_rf = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_rf.pkl'))  # noqa
        file_test_model_stacker_resample_means = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_means.pkl'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_cart,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='cart'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_rf,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='random_forest'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_means,
                                                     expected_dataframe=fitter.model.get_resample_means())

        if OOLearningHelpers.is_series_numeric(data[target_variable]):
            file_test_model_stacker_train_meta_correlations = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_train_meta_correlations.pkl'))  # noqa
            file_plot_correlations = 'data/test_ModelWrappers/test_modelModel_stacker_correlations.png'
        else:
            file_test_model_stacker_train_meta_correlations = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_train_meta_correlations_string_target.pkl'))  # noqa
            file_plot_correlations = 'data/test_ModelWrappers/test_modelModel_stacker_correlations_string_target.png'  # noqa

        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_train_meta_correlations,
                                                     expected_dataframe=fitter.model._train_meta_correlations)

        TestHelper.check_plot(file_plot_correlations, lambda: fitter.model.plot_correlation_heatmap())

        return fit_time

    def test_ModelStacker_Classification(self):
        self.helper_test_ModelStacker_Classification(data=TestHelper.get_titanic_data(), positive_class=1)

    def test_ModelStacker_Classification_string_classes(self):
        data = TestHelper.get_titanic_data()
        positive_class = 'lived'
        negative_class = 'died'
        # Test with a string target variable rather than 0/1
        data.Survived = np.where(data.Survived == 1, positive_class, negative_class)
        self.helper_test_ModelStacker_Classification(data=data, positive_class=positive_class)

    def test_ModelStacker_Regression_no_stacker_transformations(self):
        ######################################################################################################
        # elastic_net stacker, svm & GBM & polynomial linear regression base models
        ######################################################################################################
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'

        # Use same splitter information to get the training/holdout data
        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, test_indexes = splitter.split(target_values=data[target_variable])
        expected_train_y = data.iloc[training_indexes][target_variable]
        expected_train_x = data.iloc[training_indexes].drop(columns=target_variable)
        expected_test_x = data.iloc[test_indexes].drop(columns=target_variable)

        file_train_callback = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_regression_train_callback_train_meta.pkl'))  # noqa
        file_test_callback_traindata = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_regression_test_callback_test_meta_traindata.pkl'))  # noqa
        file_test_callback_holdoutdata = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_regression_test_callback_test_meta_holdoutdata.pkl'))  # noqa

        # these variables help us verify the callbacks were called
        train_callback_called = list()
        predict_callback_called = list()

        def train_callback(train_meta_x, train_meta_y, hyper_params):
            # the `train_meta_x` that we built in `train()` should have the same indexes as `train_meta_x
            assert all(train_meta_x.index.values == expected_train_x.index.values)
            # cached dataframe in `file_train_callback` are the predictions from the base model we expect
            TestHelper.ensure_all_values_equal_from_file(file=file_train_callback,
                                                         expected_dataframe=train_meta_x)
            assert all(train_meta_y.index.values == expected_train_x.index.values)
            # check that the y values in `train_meta_y` which will be trained on match the y's from the split
            assert all(train_meta_y == expected_train_y)
            assert isinstance(hyper_params, ElasticNetRegressorHP)

            # add value to the list so we know this callback was called and completed
            train_callback_called.append('train_called')

        # predict will be called twice, first for evaluating the training data; next for the holdout
        def predict_callback(test_meta):
            if len(predict_callback_called) == 0:  # first time called i.e. training evaluation
                # the `test_meta` that we built in `predict()` should have the same indexes as `train_meta_x`
                # when evaluating the training set
                assert all(test_meta.index.values == expected_train_x.index.values)
                # cached dataframe in `file_test_callback_traindata` are the predictions from the base
                # models (refitted on the entire training set) on the dataset that was passed into `predict()`
                # in this case the original training data, which will then be predicted on the final stacking
                # model
                TestHelper.ensure_all_values_equal_from_file(file=file_test_callback_traindata,
                                                             expected_dataframe=test_meta)
                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_train')

            elif len(predict_callback_called) == 1:  # second time called i.e. holdout evaluation
                # the `test_meta` that we built in `predict()` should have the same indexes as
                # `expected_test_x` when evaluating the holdout set
                assert all(test_meta.index.values == expected_test_x.index.values)
                # cached dataframe in `file_test_callback_holdoutdata` are the predictions from the base
                # models (refitted on the entire training set) on the dataset that was passed into `predict()`
                # in this case the holdout data, which will then be predicted on the final stacking model
                TestHelper.ensure_all_values_equal_from_file(file=file_test_callback_holdoutdata,
                                                             expected_dataframe=test_meta)

                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_holdout')
            else:
                raise ValueError()

        model_stacker = ModelStacker(base_models=[ModelDefaults.get_LinearRegressor(degrees=2),
                                                  ModelDefaults.get_CartDecisionTreeRegressor(),
                                                  ModelDefaults.get_GradientBoostingRegressor()],
                                     scores=[MaeScore(), RmseScore()],
                                     stacking_model=ElasticNetRegressor(),
                                     stacking_transformations=None,
                                     train_callback=train_callback,
                                     predict_callback=predict_callback)

        # use a fitter so we get don't have to worry about splitting/transforming/evaluating/etc.
        fitter = ModelTrainer(model=model_stacker,
                              model_transformations=None,  # transformed for stacker and base models.
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        assert len(train_callback_called) == 0
        assert len(predict_callback_called) == 0
        fitter.train(data=data, target_variable=target_variable, hyper_params=ElasticNetRegressorHP())
        # verify our callback is called. If it wasn't, we would never know and the assertions wouldn't run.
        assert train_callback_called == ['train_called']
        # `predict_callback` should be called TWICE (once for training eval & once for holdout eval)
        assert predict_callback_called == ['predict_called_train', 'predict_called_holdout']

        expected_training_evaluator_metrics = {'Mean Absolute Error (MAE)': 1986.2912789878249, 'Mean Squared Error (MSE)': 13417752.365422385, 'Root Mean Squared Error (RMSE)': 3663.0250293196723, 'RMSE to Standard Deviation of Target': 0.3043811875255302, 'Total Observations': 1070}  # noqa
        assert all([isclose(expected_training_evaluator_metrics[key], fitter.training_evaluator.all_quality_metrics[key]) for key in fitter.training_evaluator.all_quality_metrics.keys()])  # noqa
        expected_holdout_evaluator_metrics = {'Mean Absolute Error (MAE)': 2540.1233867075684, 'Mean Squared Error (MSE)': 22559042.48804891, 'Root Mean Squared Error (RMSE)': 4749.63603742949, 'RMSE to Standard Deviation of Target': 0.3835145563392683, 'Total Observations': 268}  # noqa
        assert all([isclose(expected_holdout_evaluator_metrics[key], fitter.holdout_evaluator.all_quality_metrics[key]) for key in fitter.training_evaluator.all_quality_metrics.keys()])  # noqa

        assert [x.description for x in model_stacker._base_models] == ['LinearRegressor_polynomial_2', 'CartDecisionTreeRegressor', 'GradientBoostingRegressor']  # noqa
        # test resample data
        file_test_model_stacker_resample_data_regression = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_regression.pkl'))  # noqa
        file_test_model_stacker_resample_data_cart = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_cart_regression.pkl'))  # noqa
        file_test_model_stacker_resample_data_gb = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_gb.pkl'))  # noqa
        file_test_model_stacker_resample_means = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_means_regression.pkl'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_regression,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='LinearRegressor_polynomial_2'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_cart,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='CartDecisionTreeRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_gb,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='GradientBoostingRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_means,
                                                     expected_dataframe=fitter.model.get_resample_means())

        file_test_model_stacker_train_meta_correlations = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_train_meta_correlations_regression.pkl'))  # noqa
        file_plot_correlations = 'data/test_ModelWrappers/test_modelModel_stacker_correlations_regression.png'
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_train_meta_correlations,
                                                     expected_dataframe=fitter.model._train_meta_correlations)
        TestHelper.check_plot(file_plot_correlations, lambda: fitter.model.plot_correlation_heatmap())

    def test_ModelStacker_Regression_with_stacker_transformations(self):
        ######################################################################################################
        # elastic_net stacker, svm & GBM & polynomial linear regression base models
        ######################################################################################################
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'

        # Use same splitter information to get the training/holdout data
        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, test_indexes = splitter.split(target_values=data[target_variable])
        expected_train_y = data.iloc[training_indexes][target_variable]
        expected_train_x = data.iloc[training_indexes].drop(columns=target_variable)
        expected_test_x = data.iloc[test_indexes].drop(columns=target_variable)

        file_train_callback = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_regression_train_callback_train_meta_stacker_trans.pkl'))  # noqa
        file_test_callback_traindata = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_regression_test_callback_test_meta_traindata_stacker_trans.pkl'))  # noqa
        file_test_callback_holdoutdata = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_regression_test_callback_test_meta_holdoutdata_stacker_trans.pkl'))  # noqa

        # these variables help us verify the callbacks were called
        train_callback_called = list()
        predict_callback_called = list()

        def train_callback(train_meta_x, train_meta_y, hyper_params):
            # the `train_meta_x` that we built in `train()` should have the same indexes as `train_meta_x
            assert all(train_meta_x.index.values == expected_train_x.index.values)
            # cached dataframe in `file_train_callback` are the predictions from the base model we expect
            TestHelper.ensure_all_values_equal_from_file(file=file_train_callback,
                                                         expected_dataframe=train_meta_x)
            assert all(train_meta_y.index.values == expected_train_x.index.values)
            # check that the y values in `train_meta_y` which will be trained on match the y's from the split
            assert all(train_meta_y == expected_train_y)
            assert isinstance(hyper_params, ElasticNetRegressorHP)

            # add value to the list so we know this callback was called and completed
            train_callback_called.append('train_called')

        # predict will be called twice, first for evaluating the training data; next for the holdout
        def predict_callback(test_meta):
            if len(predict_callback_called) == 0:  # first time called i.e. training evaluation
                # the `test_meta` that we built in `predict()` should have the same indexes as `train_meta_x`
                # when evaluating the training set
                assert all(test_meta.index.values == expected_train_x.index.values)
                # cached dataframe in `file_test_callback_traindata` are the predictions from the base
                # models (refitted on the entire training set) on the dataset that was passed into `predict()`
                # in this case the original training data, which will then be predicted on the final stacking
                # model
                TestHelper.ensure_all_values_equal_from_file(file=file_test_callback_traindata,
                                                             expected_dataframe=test_meta)
                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_train')

            elif len(predict_callback_called) == 1:  # second time called i.e. holdout evaluation
                # the `test_meta` that we built in `predict()` should have the same indexes as
                # `expected_test_x` when evaluating the holdout set
                assert all(test_meta.index.values == expected_test_x.index.values)
                # cached dataframe in `file_test_callback_holdoutdata` are the predictions from the base
                # models (refitted on the entire training set) on the dataset that was passed into `predict()`
                # in this case the holdout data, which will then be predicted on the final stacking model
                TestHelper.ensure_all_values_equal_from_file(file=file_test_callback_holdoutdata,
                                                             expected_dataframe=test_meta)

                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_holdout')
            else:
                raise ValueError()

        model_stacker = ModelStacker(base_models=[ModelDefaults.get_LinearRegressor(degrees=2),
                                                  ModelDefaults.get_CartDecisionTreeRegressor(),
                                                  ModelDefaults.get_GradientBoostingRegressor()],
                                     scores=[MaeScore(), RmseScore()],
                                     stacking_model=ElasticNetRegressor(),
                                     stacking_transformations=[CenterScaleTransformer()],
                                     train_callback=train_callback,
                                     predict_callback=predict_callback)

        # use a fitter so we get don't have to worry about splitting/transforming/evaluating/etc.
        fitter = ModelTrainer(model=model_stacker,
                              # transformed for stacker and base models.
                              model_transformations=None,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        assert len(train_callback_called) == 0
        assert len(predict_callback_called) == 0
        fitter.train(data=data, target_variable=target_variable, hyper_params=ElasticNetRegressorHP())
        # verify our callback is called. If it wasn't, we would never know and the assertions wouldn't run.
        assert train_callback_called == ['train_called']
        # `predict_callback` should be called TWICE (once for training eval & once for holdout eval)
        assert predict_callback_called == ['predict_called_train', 'predict_called_holdout']

        expected_training_evaluator_metrics = {'Mean Absolute Error (MAE)': 2352.581933713817, 'Mean Squared Error (MSE)': 12286239.244358376, 'Root Mean Squared Error (RMSE)': 3505.173211748369, 'RMSE to Standard Deviation of Target': 0.29126439926969333, 'Total Observations': 1070}  # noqa
        assert all([isclose(expected_training_evaluator_metrics[key], fitter.training_evaluator.all_quality_metrics[key]) for key in fitter.training_evaluator.all_quality_metrics.keys()])  # noqa
        expected_holdout_evaluator_metrics = {'Mean Absolute Error (MAE)': 3130.3878123286368, 'Mean Squared Error (MSE)': 25683778.713447947, 'Root Mean Squared Error (RMSE)': 5067.916604823717, 'RMSE to Standard Deviation of Target': 0.40921446884491564, 'Total Observations': 268}  # noqa
        assert all([isclose(expected_holdout_evaluator_metrics[key], fitter.holdout_evaluator.all_quality_metrics[key]) for key in fitter.training_evaluator.all_quality_metrics.keys()])  # noqa
        assert all([x == y for x, y in zip([x.description for x in model_stacker._base_models], ['LinearRegressor_polynomial_2', 'CartDecisionTreeRegressor', 'GradientBoostingRegressor'])])  # noqa
        # test resample data
        file_test_model_stacker_resample_data_regression = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_regression_stacker_trans.pkl'))  # noqa
        file_test_model_stacker_resample_data_cart = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_cart_regression_stacker_trans.pkl'))  # noqa
        file_test_model_stacker_resample_data_gb = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_data_gb_stacker_trans.pkl'))  # noqa
        file_test_model_stacker_resample_means = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_resample_means_regression_stacker_trans.pkl'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_regression,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='LinearRegressor_polynomial_2'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_cart,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='CartDecisionTreeRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_gb,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='GradientBoostingRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_means,
                                                     expected_dataframe=fitter.model.get_resample_means())

        file_test_model_stacker_train_meta_correlations = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_train_meta_correlations_regression_stacker_trans.pkl'))  # noqa
        file_plot_correlations = 'data/test_ModelWrappers/test_modelModel_stacker_correlations_regression_stacker_trans.png'  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_train_meta_correlations,
                                                     expected_dataframe=fitter.model._train_meta_correlations)
        TestHelper.check_plot(file_plot_correlations, lambda: fitter.model.plot_correlation_heatmap())

    def test_ModelStacker_caching(self):
        cache_directory = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification')  # noqa

        if os.path.isdir(cache_directory):
            shutil.rmtree(cache_directory)
        assert not os.path.isdir(cache_directory)

        # cache files do not exist
        fit_time_not_previously_cached = self.helper_test_ModelStacker_Classification(data=TestHelper.get_titanic_data(),  # noqa
                                                                                      positive_class=1,
                                                                                      cache_directory=cache_directory)  # noqa

        fit_time_previously_cached = self.helper_test_ModelStacker_Classification(data=TestHelper.get_titanic_data(),  # noqa
                                                                                  positive_class=1,
                                                                                  cache_directory=cache_directory)  # noqa
        shutil.rmtree(cache_directory)
        assert 6 < fit_time_not_previously_cached < 8  # looks like around ~7 seconds on average
        assert fit_time_previously_cached < 2  # improves to less than 2 with caching
