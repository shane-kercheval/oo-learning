import copy
import os
import os.path
import pickle
import shutil
import time
import warnings
from math import isclose
from typing import Callable

import numpy as np
import sklearn as sk

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

    def flush(self): pass


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

    # noinspection PyPep8Naming
    @staticmethod
    def get_GradientBoostingRegressor() -> ModelInfo:
        model_wrapper = GradientBoostingRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         # TODO: fill out rest of recommended transformations, verify order
                         transformations=[ImputationTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=GradientBoostingRegressorHP(),
                         hyper_params_grid=dict(learning_rate=[0.1, 0.5, 1],
                                                n_estimators=[50, 100, 5000],
                                                max_depth=[1, 5, 9],
                                                min_samples_leaf=[1, 10, 20]))

    @staticmethod
    def get_CartDecisionTreeRegressor() -> ModelInfo:
        model_wrapper = CartDecisionTreeRegressor()
        return ModelInfo(description=type(model_wrapper).__name__,
                         model=model_wrapper,
                         transformations=[ImputationTransformer(),
                                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                         hyper_params=CartDecisionTreeHP(criterion='mse'),
                         hyper_params_grid=dict(max_depth=[3, 10, 30]))

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

    def test_OOLearningHelpers_get_final_datasets(self):
        data = TestHelper.get_titanic_data()

        ######################################################################################################
        # test without Splitter
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        t_x, t_y, h_x, h_y, pipeline = OOLearningHelpers.get_final_datasets(data=data,
                                                                            target_variable='Survived',
                                                                            splitter=None,
                                                                            transformations=transformations)
        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_OOLearningHelpers_get_final_datasets_no_splitter_train_x.pkl'),  # noqa
                                                     expected_dataframe=t_x)
        assert list(t_y) == [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]  # noqa
        assert all(t_y.index.values == t_x.index.values)
        assert h_x.shape == (0, 11)
        assert all(h_x.columns.values == ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])  # noqa
        assert len(h_y) == 0
        assert all(h_y.index.values == h_x.index.values)
        assert [x.state for x in pipeline.transformations] == [{}, {'Pclass': [1, 2, 3], 'SibSp': [0, 1, 2, 3, 4, 5, 8], 'Parch': [0, 1, 2, 3, 4, 5, 6]}, {'Age': 28.0, 'Fare': 14.4542, 'Pclass': 3, 'Sex': 'male', 'SibSp': 0, 'Parch': 0, 'Embarked': 'S'}, {'Pclass': [1, 2, 3], 'Sex': ['female', 'male'], 'SibSp': [0, 1, 2, 3, 4, 5, 8], 'Parch': [0, 1, 2, 3, 4, 5, 6], 'Embarked': ['C', 'Q', 'S']}, {}]  # noqa

        ######################################################################################################
        # test with Splitter
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        t_x, t_y, h_x, h_y, pipeline = OOLearningHelpers.get_final_datasets(data=data,
                                                                            target_variable='Survived',
                                                                            splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),  # noqa
                                                                            transformations=transformations)
        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_OOLearningHelpers_get_final_datasets_with_splitter_train_x.pkl'),  # noqa
                                                     expected_dataframe=t_x)
        assert list(t_y) == [1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]  # noqa
        assert all(t_y.index.values == t_x.index.values)
        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_ModelWrappers/test_OOLearningHelpers_get_final_datasets_with_splitter_holdout_x.pkl'),  # noqa
                                                     expected_dataframe=h_x)

        assert list(h_y) == [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]  # noqa
        assert all(h_y.index.values == h_x.index.values)
        assert [x.state for x in pipeline.transformations] == [{}, {'Pclass': [1, 2, 3], 'SibSp': [0, 1, 2, 3, 4, 5, 8], 'Parch': [0, 1, 2, 3, 4, 5, 6]}, {'Age': 28.5, 'Fare': 14.4542, 'Pclass': 3, 'Sex': 'male', 'SibSp': 0, 'Parch': 0, 'Embarked': 'S'}, {'Pclass': [1, 2, 3], 'Sex': ['female', 'male'], 'SibSp': [0, 1, 2, 3, 4, 5, 8], 'Parch': [0, 1, 2, 3, 4, 5, 6], 'Embarked': ['C', 'Q', 'S']}, {}]  # noqa

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
        # calling `set_persistence_manager()` after `train_predict_eval()` should fail
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
        # predicting with a cached model that does not exist (need to call `train_predict_eval()` before
        # `predict()`)
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
        # predicting with a cached model that already exists (still need to call `train_predict_eval()`
        # before `predict()`, because train_predict_eval has parameters that are needed to set additional info
        # after model has been fitted. `predict()` should not change, basically testing that we have a model
        # via model_object we already tested above that the correct model_object is being cached/retrieved
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
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=RandomForestHP(criterion='gini'))

    def test_Regression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelTrainer(model=LinearRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=None)

        fitter.train_predict_eval(data=data, target_variable=target_variable, hyper_params=None)

        TestHelper.save_string(fitter, 'data/test_ModelWrappers/test_Regression_string.txt')

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert isclose(fitter.training_evaluator.mean_squared_error, 109.68243774089586)
        assert isclose(fitter.training_evaluator.mean_absolute_error, 8.360259532214116)
        assert isclose(fitter.holdout_evaluator.mean_squared_error, 100.07028301004217)
        assert isclose(fitter.holdout_evaluator.mean_absolute_error, 7.99161252047238)

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

        hyper_params = RidgeRegressorHP(alpha=0)
        holdout_predictions = fitter.train_predict_eval(data=data,
                                                        target_variable=target_variable,
                                                        hyper_params=hyper_params)

        TestHelper.assert_hyper_params_match(hyper_params, fitter.model)
        assert fitter.model.hyper_params.params_dict == {'alpha': 0, 'solver': 'cholesky'}
        TestHelper.save_string(fitter, 'data/test_ModelWrappers/test_RidgeRegression_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_RidgeRegression_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_RidgeRegression_holdout_evaluator.txt')

        # because we supplied a Splitter, we will get the holdout-predictions
        # check to make sure it gives the same MAE
        assert len(holdout_predictions) == len(data) * 0.20
        _, holdout_indexes = RegressionStratifiedDataSplitter(holdout_ratio=0.20).split(target_values=data[target_variable])  # noqa
        manual_score = MaeScore().calculate(actual_values=data.iloc[holdout_indexes][target_variable].values,
                                            predicted_values=holdout_predictions)
        assert isclose(manual_score, fitter.holdout_evaluator.mean_absolute_error)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'age']  # noqa

        # Test that tuner works with hyper-params
        # I also ran a Resampler with LinearRegressor model and made sure it had the same values as
        # as the Tuner.Resampler with Alpha == 0
        train_data_y = data[target_variable]
        train_data = data.drop(columns=target_variable)
        evaluators = [MaeScore(), RmseScore()]
        grid = HyperParamsGrid(params_dict={'alpha': [0, 0.5, 1]})
        tuner = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=RidgeRegressor(),
                                                                                transformations=[RemoveColumnsTransformer(columns=['fineagg']),  # noqa
                                                                                       CenterScaleTransformer()],  # noqa
                                                                                scores=evaluators),
                                     hyper_param_object=RidgeRegressorHP(),
                                     params_grid=grid)
        tuner.tune(data_x=train_data, data_y=train_data_y)

        assert len(tuner.results._tune_results_objects) == 3
        assert tuner.results.number_of_cycles == 3

        TestHelper.save_string(tuner.results,
                               'data/test_ModelWrappers/test_RidgeRegressionr_tuner_results.txt')

    def test_LassoRegression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelTrainer(model=LassoRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=None)

        hyper_params = LassoRegressorHP(alpha=0)
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=hyper_params)
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert fitter.model.hyper_params.params_dict == {'alpha': 0}
        TestHelper.assert_hyper_params_match(hyper_params, fitter.model)
        TestHelper.save_string(fitter, 'data/test_ModelWrappers/test_LassoRegression_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_LassoRegression_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_LassoRegression_holdout_evaluator.txt')

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'age']  # noqa

        # Test that tuner works with hyper-params
        # I also ran a Resampler with LinearRegressor model and made sure it had the same values as
        # as the Tuner.Resampler with Alpha == 0
        train_data_y = data[target_variable]
        train_data = data.drop(columns=target_variable)
        evaluators = [MaeScore(), RmseScore()]
        grid = HyperParamsGrid(params_dict={'alpha': [0.1, 0.5, 1]})
        tuner = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=LassoRegressor(),
                                                                                transformations=[RemoveColumnsTransformer(columns=['fineagg']),  # noqa
                                                                                       CenterScaleTransformer()],  # noqa
                                                                                scores=evaluators),
                                     hyper_param_object=LassoRegressorHP(),
                                     params_grid=grid,
                                     parallelization_cores=-1)
        tuner.tune(data_x=train_data, data_y=train_data_y)
        assert len(tuner.results._tune_results_objects) == 3
        assert tuner.results.number_of_cycles == 3

        TestHelper.save_string(tuner.results,
                               'data/test_ModelWrappers/test_LassoRegression_tuner_results.txt')

    def test_ElasticNetRegressor(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        fitter = ModelTrainer(model=ElasticNetRegressor(),
                              model_transformations=[RemoveColumnsTransformer(columns=['fineagg'])],
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                              evaluator=RegressionEvaluator(),
                              persistence_manager=None,
                              train_callback=None)

        hyper_params = ElasticNetRegressorHP(alpha=0)
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=hyper_params)
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)
        assert fitter.model.hyper_params.params_dict == {'alpha': 0, 'l1_ratio': 0.5}

        TestHelper.assert_hyper_params_match(hyper_params, fitter.model)
        TestHelper.save_string(fitter, 'data/test_ModelWrappers/test_ElasticNetRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_ElasticNetRegressor_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_ElasticNetRegressor_holdout_evaluator.txt')

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'age']  # noqa

        # Test that tuner works with hyper-params
        # I also ran a Resampler with LinearRegressor model and made sure it had the same values as
        # as the Tuner.Resampler with Alpha == 0
        train_data_y = data[target_variable]
        train_data = data.drop(columns=target_variable)
        evaluators = [MaeScore(), RmseScore()]
        grid = HyperParamsGrid(params_dict={'alpha': [0.1, 0.5, 1], 'l1_ratio': [0.2, 0.6]})
        tuner = GridSearchModelTuner(resampler=RepeatedCrossValidationResampler(model=ElasticNetRegressor(),
                                                                                transformations=[RemoveColumnsTransformer(columns=['fineagg']),  # noqa
                                                                                       CenterScaleTransformer()],  # noqa
                                                                                scores=evaluators),
                                     hyper_param_object=ElasticNetRegressorHP(),
                                     params_grid=grid,
                                     parallelization_cores=-1)
        tuner.tune(data_x=train_data, data_y=train_data_y)
        assert len(tuner.results._tune_results_objects) == 6
        assert tuner.results.number_of_cycles == 6

        TestHelper.save_string(tuner.results,
                               'data/test_ModelWrappers/test_ElasticNetRegressor_tuner_results.txt')

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
        self.assertRaises(NotImplementedError,
                          lambda: model_fitter.train_predict_eval(data=data,
                                                                  target_variable=target_variable,
                                                                  hyper_params=None))

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
        # make sure the data that we pass to `train_predict_eval()` in the ModelWrapper is transformed
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
        # train_predict_eval/predict the model using the Mock object, which stores the transformed
        # training/test data so we can validate the expected transformations took place across both datasets
        ######################################################################################################
        model_fitter = ModelTrainer(model=MockRegressionModelWrapper(data_y=data.strength),
                                    model_transformations=[RemoveColumnsTransformer(['coarseagg', 'fineagg']),
                                                           ImputationTransformer(),
                                                           DummyEncodeTransformer()],
                                    evaluator=RegressionEvaluator(),
                                    splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20),
                                    train_callback=train_callback)

        # should raise an error calling `predict` before `train_predict_eval`
        self.assertRaises(ModelNotFittedError, lambda: model_fitter.predict(data_x=test_x))

        model_fitter.train_predict_eval(data=data, target_variable=target_variable, hyper_params=None)

        assert isclose(model_fitter.training_evaluator.root_mean_squared_error, 24.50861705505752)
        assert isclose(model_fitter.training_evaluator.mean_absolute_error, 19.700946601941748)

        assert isclose(model_fitter.holdout_evaluator.root_mean_squared_error, 23.528246193289437)
        assert isclose(model_fitter.holdout_evaluator.mean_absolute_error, 19.254368932038837)

        # should not be able to call train_predict_eval twice
        self.assertRaises(ModelAlreadyFittedError,
                          lambda: model_fitter.train_predict_eval(data=data,
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
        # "modify ModelTrainer to *optionally* take a Splitter, and train_predict_eval on all data"
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
        holdout_predictions = fitter.train_predict_eval(data=data,
                                                        target_variable=target_variable,
                                                        hyper_params=None)
        # because we supplied a Splitter, we will get the holdout-predictions
        # check to make sure it gives the same MAE
        assert len(holdout_predictions) == len(data) * 0.20
        _, holdout_indexes = RegressionStratifiedDataSplitter(holdout_ratio=0.20).split(
            target_values=data[target_variable])
        manual_score = MaeScore().calculate(actual_values=data.iloc[holdout_indexes][target_variable].values,
                                            predicted_values=holdout_predictions)
        assert isclose(manual_score, fitter.holdout_evaluator.mean_absolute_error)

        assert len(train_callback_called) == 1
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        expected_dictionary = {'Mean Absolute Error (MAE)': 8.239183296671666, 'Mean Squared Error (MSE)': 105.83192307350386, 'Root Mean Squared Error (RMSE)': 10.287464365600682, 'RMSE to Standard Deviation of Target': 0.6194779398676552, 'R Squared': 0.6162470820173258, 'Total Observations': 824}  # noqa
        TestHelper.ensure_values_numeric_dictionary(dictionary_1=expected_dictionary,
                                                    dictionary_2=fitter.training_evaluator.all_quality_metrics)  # noqa

        expected_dictionary = {'Mean Absolute Error (MAE)': 8.381063270249973, 'Mean Squared Error (MSE)': 114.99391155644095, 'Root Mean Squared Error (RMSE)': 10.723521415861534, 'RMSE to Standard Deviation of Target': 0.6287320504285148, 'R Squared': 0.6046960087639556, 'Total Observations': 206}  # noqa
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
        training_predictions = fitter.train_predict_eval(data=data,
                                                         target_variable=target_variable,
                                                         hyper_params=None)
        # because we didn't supply a Splitter, we will get the training-predictions
        # check to make sure it gives the same MAE
        assert len(training_predictions) == len(data)  # no splitter, used all data
        manual_score = MaeScore().calculate(actual_values=data[target_variable].values,
                                            predicted_values=training_predictions)
        assert isclose(manual_score, fitter.training_evaluator.mean_absolute_error)

        assert len(train_callback_called) == 1
        # if we don't have a splitter, we can still have a training evaluator, but a holdout evaluator (no
        # holdout to evaluate)
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert fitter.holdout_evaluator is None
        assert fitter.holdout_scores is None
        # ensure the number of observations in the Training Evaluator matches the number of observations in
        # the entire dataset.
        expected_dictionary = {'Mean Absolute Error (MAE)': 8.264694852478934, 'Mean Squared Error (MSE)': 107.57095170273888, 'Root Mean Squared Error (RMSE)': 10.371641707210044, 'RMSE to Standard Deviation of Target': 0.6211445248863783, 'R Squared': 0.6141794792036753, 'Total Observations': 1030}  # noqa
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
            hp = LogisticClassifierHP()
            fitter.train_predict_eval(data=data,
                                      target_variable='Survived',
                                      hyper_params=hp)
            assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
            assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

            TestHelper.assert_hyper_params_match(hyper_params=hp, model=fitter.model,
                                                 mapping={'regularization_inverse': 'C'})
            TestHelper.save_string(fitter, 'data/test_ModelWrappers/test_Logistic_string.txt')
            TestHelper.save_string(fitter.training_evaluator,
                                   'data/test_ModelWrappers/test_Logistic_training_evaluator.txt')
            TestHelper.save_string(fitter.holdout_evaluator,
                                   'data/test_ModelWrappers/test_Logistic_holdout_evaluator.txt')

            TestHelper.check_plot(file_name='data/test_ModelWrappers/test_LogisticMW_training_ROC.png',
                                  get_plot_function=lambda: fitter.training_evaluator.plot_roc_curve(),
                                  set_size_w_h=(8, 8))
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_training_PrecRecal.png',
                                  lambda: fitter.training_evaluator.plot_ppv_tpr_curve(),
                                  set_size_w_h=(8, 8))

            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_holdout_ROC.png',
                                  lambda: fitter.holdout_evaluator.plot_roc_curve(),
                                  set_size_w_h=(8, 8))
            TestHelper.check_plot('data/test_ModelWrappers/test_LogisticMW_holdout_PrecRecal.png',
                                  lambda: fitter.holdout_evaluator.plot_ppv_tpr_curve(),
                                  set_size_w_h=(8, 8))

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
        hp = LogisticClassifierHP()
        fitter.train_predict_eval(data=data, target_variable='Survived', hyper_params=hp)
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'regularization_inverse': 'C'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_LogisticMW_string_target_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_LogisticMW_string_target_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_LogisticMW_string_target_holdout_evaluator.txt')

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
        hp = RandomForestHP(criterion='gini')
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert os.path.isfile(fitter._persistence_manager._cache_path)
        assert isinstance(fitter.model.model_object, sk.ensemble.RandomForestClassifier)

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
        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_RandomForestClassifier_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier_holdout_evaluator.txt')

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
        hp = RandomForestHP(criterion='gini', max_features='auto', n_estimators=10)
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert os.path.isfile(fitter._persistence_manager._cache_path)
        assert isinstance(fitter.model.model_object, sk.ensemble.RandomForestClassifier)

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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_RandomForestClassifier2_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier2_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier2_holdout_evaluator.txt')

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

        hp = RandomForestHP(criterion='gini')
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.RandomForestClassifier)

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
        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_RandomForestClassifier_string_target_string.txt')  # noqa
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier_string_target_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier_string_target_holdout_evaluator.txt')  # noqa

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

        hp = RandomForestHP(criterion='gini')
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.RandomForestClassifier)

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_RandomForestClassifier_scores_string.txt')

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
        hp = RandomForestHP(criterion='MAE', n_estimators=10)
        fitter.train_predict_eval(data=data, target_variable='strength', hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.RandomForestRegressor)

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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_RandomForestRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_RandomForestRegressor_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_RandomForestRegressor_holdout_evaluator.txt')

    def test_RandomForestClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=RandomForestClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))

        hp = RandomForestHP(criterion='gini', n_estimators=10, max_features='auto')
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.RandomForestClassifier)

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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_RandomForestClassifier_multiclass_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier_multiclass_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_RandomForestClassifier_multiclass_holdout_evaluator.txt')  # noqa

    def test_ExtraTreesClassifier(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=RandomForestClassifier(extra_trees_implementation=True),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)))

        hp = RandomForestHP(criterion='gini')
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.ExtraTreesClassifier)

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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_holdout_evaluator.txt')

    def test_ExtraTreesClassifier_string_target(self):
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
        fitter = ModelTrainer(model=RandomForestClassifier(extra_trees_implementation=True),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=positive_class)))

        hp = RandomForestHP(criterion='gini')
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.ExtraTreesClassifier)

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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_string_target_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_string_target_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_string_target_holdout_evaluator.txt')  # noqa

    def test_ExtraTreesRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = [ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=RandomForestRegressor(extra_trees_implementation=True),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        hp = RandomForestHP(criterion='MAE', n_estimators=10)
        fitter.train_predict_eval(data=data,
                                  target_variable='strength',
                                  hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.ExtraTreesRegressor)
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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_ExtraTreesRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesRegressor_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesRegressor_holdout_evaluator.txt')

    def test_ExtraTreesClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=RandomForestClassifier(extra_trees_implementation=True),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))

        hp = RandomForestHP(criterion='gini', n_estimators=10, max_features='auto')
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=hp)
        assert isinstance(fitter.model.model_object, sk.ensemble.ExtraTreesClassifier)
        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_multiclass_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_multiclass_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_ExtraTreesClassifier_multiclass_holdout_evaluator.txt')  # noqa

    def test_SoftmaxRegression_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=SoftmaxLogisticClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        hp = SoftmaxLogisticHP()
        fitter.train_predict_eval(data=data, target_variable=target_variable, hyper_params=hp)

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length',
                                                   'petal_width']  # noqa
        assert fitter.model.hyper_params.params_dict == {'regularization_inverse': 1.0, 'solver': 'lbfgs'}  # noqa

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'regularization_inverse': 'C'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SoftmaxRegression_multiclass_string.txt')  # noqa
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SoftmaxRegression_multiclass_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SoftmaxRegression_multiclass_holdout_evaluator.txt')  # noqa

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
        fitter.train_predict_eval(data=data, target_variable=target_variable)

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
        fitter.train_predict_eval(data=data, target_variable=target_variable)

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

        ######################################################################################################
        # No class weights
        ######################################################################################################
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
        hp = SvmLinearClassifierHP()
        fitter.train_predict_eval(data=data, target_variable='Survived', hyper_params=hp)
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model._class_weights is None

        assert fitter.model.hyper_params.params_dict == {'penalty': 'l2', 'penalty_c': 1.0, 'loss': 'hinge'}

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'penalty': 'base_estimator__penalty',
                                                                        'penalty_c': 'base_estimator__C',
                                                                        'loss': 'base_estimator__loss'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SVM_classification_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_holdout_evaluator.txt')

        ######################################################################################################
        # Class Weights
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=SvmLinearClassifier(class_weights={0: 0.3, 1: 0.7}),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)))
        hp = SvmLinearClassifierHP()
        fitter.train_predict_eval(data=data, target_variable='Survived', hyper_params=hp)
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model._class_weights == {0: 0.3, 1: 0.7}
        assert fitter.model.hyper_params.params_dict == {'penalty': 'l2', 'penalty_c': 1.0, 'loss': 'hinge'}

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'penalty': 'base_estimator__penalty',
                                                                        'penalty_c': 'base_estimator__C',
                                                                        'loss': 'base_estimator__loss'})

        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SVM_classification_class_weights_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_class_weights_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_class_weights_holdout_evaluator.txt')  # noqa

    def test_SVM_classification_string(self):
        data = TestHelper.get_titanic_data()
        positive_class = 'lived'
        negative_class = 'died'
        # Test with a string target variable rather than 0/1
        data.Survived = np.where(data.Survived == 1, positive_class, negative_class)
        ######################################################################################################
        # No class weights
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=SvmLinearClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5, positive_class='lived')))
        hp = SvmLinearClassifierHP()
        fitter.train_predict_eval(data=data, target_variable='Survived', hyper_params=hp)
        assert fitter.model._class_weights is None
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.hyper_params.params_dict == {'penalty': 'l2', 'penalty_c': 1.0, 'loss': 'hinge'}
        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'penalty': 'base_estimator__penalty',
                                                                        'penalty_c': 'base_estimator__C',
                                                                        'loss': 'base_estimator__loss'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SVM_classification_string_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_string_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_string_holdout_evaluator.txt')  # noqa

        ######################################################################################################
        # Class class weights
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=SvmLinearClassifier(class_weights={'died': 0.3, 'lived': 0.7}),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5, positive_class='lived')))
        hp = SvmLinearClassifierHP()
        fitter.train_predict_eval(data=data, target_variable='Survived', hyper_params=hp)
        assert fitter.model._class_weights == {'died': 0.3, 'lived': 0.7}
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.hyper_params.params_dict == {'penalty': 'l2', 'penalty_c': 1.0, 'loss': 'hinge'}
        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'penalty': 'base_estimator__penalty',
                                                                        'penalty_c': 'base_estimator__C',
                                                                        'loss': 'base_estimator__loss'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SVM_classification_string_class_weights_string.txt')  # noqa
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_string_class_weights_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SVM_classification_string_class_weights_holdout_evaluator.txt')  # noqa

    def test_SvmPolynomial_classification(self):
        data = TestHelper.get_titanic_data()

        ######################################################################################################
        # No class weights
        ######################################################################################################
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
        hp = SvmPolynomialClassifierHP()
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert fitter.model._class_weights is None
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.hyper_params.params_dict == {'degree': 3, 'coef0': 0.0, 'penalty_c': 1.0}

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'degree': 'base_estimator__degree',
                                                                        'coef0': 'base_estimator__coef0',
                                                                        'penalty_c': 'base_estimator__C'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SvmPolynomial_classification_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SvmPolynomial_classification_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SvmPolynomial_classification_holdout_evaluator.txt')  # noqa

        ######################################################################################################
        # class weights
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           CenterScaleTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        # test with custom threshold of 0.5
        fitter = ModelTrainer(model=SvmPolynomialClassifier(class_weights={0: 0.3, 1: 0.7}),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1)))
        hp = SvmPolynomialClassifierHP()
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert fitter.model._class_weights == {0: 0.3, 1: 0.7}
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)

        assert fitter.model.hyper_params.params_dict == {'degree': 3, 'coef0': 0.0, 'penalty_c': 1.0}

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'degree': 'base_estimator__degree',
                                                                        'coef0': 'base_estimator__coef0',
                                                                        'penalty_c': 'base_estimator__C'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SvmPolynomial_classification_class_weights_string.txt')  # noqa
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SvmPolynomial_classification_class_weights_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SvmPolynomial_classification_class_weights_holdout_evaluator.txt')  # noqa

    def test_SvmLinearRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=SvmLinearRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        hp = SvmLinearRegressorHP()
        fitter.train_predict_eval(data=data, target_variable='strength', hyper_params=hp)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'epsilon': 0.1, 'penalty_c': 1.0}

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'penalty_c': 'C'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SvmLinearRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SvmLinearRegressor_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SvmLinearRegressor_holdout_evaluator.txt')

    def test_SvmPolynomialRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = [ImputationTransformer(),
                           CenterScaleTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=SvmPolynomialRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        hp = SvmPolynomialRegressorHP()
        fitter.train_predict_eval(data=data,
                                  target_variable='strength',
                                  hyper_params=hp)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'degree': 3, 'epsilon': 0.1, 'penalty_c': 1.0}

        TestHelper.assert_hyper_params_match(hp, fitter.model, mapping={'penalty_c': 'C'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_SvmPolynomialRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_SvmPolynomialRegressor_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_SvmPolynomialRegressor_holdout_evaluator.txt')

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
        hp = CartDecisionTreeHP(criterion='mae')
        fitter.train_predict_eval(data=data,
                                  target_variable='strength',
                                  hyper_params=hp)

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
        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_CartDecisionTreeRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_CartDecisionTreeRegressor_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_CartDecisionTreeRegressor_holdout_evaluator.txt')  # noqa

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
        hp = CartDecisionTreeHP(criterion='gini')
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_CartDecisionTreeClassifier_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_CartDecisionTreeClassifier_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_CartDecisionTreeClassifier_holdout_evaluator.txt')  # noqa

    def test_CartDecisionTreeClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=CartDecisionTreeClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        hp = CartDecisionTreeHP(criterion='gini')
        fitter.train_predict_eval(data=data, target_variable=target_variable, hyper_params=hp)

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

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_CartDecisionTreeClassifier_multiclass_string.txt')  # noqa
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_CartDecisionTreeClassifier_multiclass_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_CartDecisionTreeClassifier_multiclass_holdout_evaluator.txt')  # noqa

    def test_AdaBoostRegressor(self):
        data = TestHelper.get_cement_data()

        fitter = ModelTrainer(model=AdaBoostRegressor(),
                              model_transformations=None,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        hp = AdaBoostRegressorHP()
        fitter.train_predict_eval(data=data, target_variable='strength', hyper_params=hp)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.hyper_params.params_dict == {'n_estimators': 50,
                                                         'learning_rate': 1.0,
                                                         'loss': 'linear',
                                                         'criterion': 'mse',
                                                         'splitter': 'best',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_features': None,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0.0}
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_AdaBoostRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_AdaBoostRegressor_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_AdaBoostRegressor_holdout_evaluator.txt')  # noqa

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
        hp = AdaBoostClassifierHP()
        fitter.train_predict_eval(data=data, target_variable='Survived', hyper_params=hp)
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'n_estimators': 50,
                                                         'learning_rate': 1.0,
                                                         'algorithm': 'SAMME.R',
                                                         'criterion': 'gini',
                                                         'splitter': 'best',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_features': None,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0.0,
                                                         'class_weight': None}

        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_AdaBoostClassifier_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_AdaBoostClassifier_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_AdaBoostClassifier_holdout_evaluator.txt')

    def test_AdaBoostClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=AdaBoostClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=AdaBoostClassifierHP())

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        assert fitter.model.hyper_params.params_dict == {'n_estimators': 50,
                                                         'learning_rate': 1.0,
                                                         'algorithm': 'SAMME.R',
                                                         'criterion': 'gini',
                                                         'splitter': 'best',
                                                         'max_depth': None,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'min_weight_fraction_leaf': 0.0,
                                                         'max_features': None,
                                                         'max_leaf_nodes': None,
                                                         'min_impurity_decrease': 0.0,
                                                         'class_weight': None}

        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_AdaBoostClassifier_multiclass_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_AdaBoostClassifier_multiclass_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_AdaBoostClassifier_multiclass_holdout_evaluator.txt')  # noqa

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
        hp = GradientBoostingRegressorHP()
        fitter.train_predict_eval(data=data,
                                  target_variable='strength',
                                  hyper_params=hp)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa
        assert fitter.model.hyper_params.params_dict == {'loss': 'ls',
                                                         'learning_rate': 0.1,
                                                         'n_estimators': 100,
                                                         'max_depth': 3,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': None,
                                                         'subsample': 1.0}

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_GradientBoostingRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_GradientBoostingRegressor_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_GradientBoostingRegressor_holdout_evaluator.txt')  # noqa

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
        hp = GradientBoostingClassifierHP()
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'loss': 'deviance',
                                                         'learning_rate': 0.1,
                                                         'n_estimators': 100,
                                                         'max_depth': 3,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': None,
                                                         'subsample': 1.0}

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_GradientBoostingClassifier_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_GradientBoostingClassifier_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_GradientBoostingClassifier_holdout_evaluator.txt')  # noqa

    def test_GradientBoostingClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=GradientBoostingClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        hp = GradientBoostingClassifierHP()
        fitter.train_predict_eval(data=data, target_variable=target_variable, hyper_params=hp)

        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)

        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']  # noqa
        assert fitter.model.hyper_params.params_dict == {'loss': 'deviance',
                                                         'learning_rate': 0.1,
                                                         'n_estimators': 100,
                                                         'max_depth': 3,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': None,
                                                         'subsample': 1.0}

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_GradientBoostingClassifier_multiclass_string.txt')  # noqa
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_GradientBoostingClassifier_multiclass_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_GradientBoostingClassifier_multiclass_holdout_evaluator.txt')  # noqa

    def test_LGBClassifier(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=LightGBMClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)))
        hp = LightGBMHP()
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa

        assert fitter.model.model_object.get_params() == {'boosting_type': 'gbdt',
                                                          'class_weight': None,
                                                          'colsample_bytree': 1.0,
                                                          'importance_type': 'split',
                                                          'learning_rate': 0.1,
                                                          'max_depth': -1,
                                                          'min_child_samples': 20,
                                                          'min_child_weight': 0.001,
                                                          'min_split_gain': 0.0,
                                                          'n_estimators': 100,
                                                          'n_jobs': -1,
                                                          'num_leaves': 31,
                                                          'objective': 'binary',
                                                          'random_state': 42,
                                                          'reg_alpha': 0.0,
                                                          'reg_lambda': 0.0,
                                                          'silent': True,
                                                          'subsample': 1.0,
                                                          'subsample_for_bin': 200000,
                                                          'subsample_freq': 0,
                                                          'scale_pos_weight': 1.0,
                                                          'max_bin': 255}  # noqa

        TestHelper.assert_hyper_params_match(hyper_params=hp, model=fitter.model,
                                             mapping={'min_gain_to_split': 'min_split_gain',
                                                      'min_sum_hessian_in_leaf': 'min_child_weight',
                                                      'min_data_in_leaf': 'min_child_samples',
                                                      'bagging_fraction': 'subsample',
                                                      'bagging_freq': 'subsample_freq',
                                                      'feature_fraction': 'colsample_bytree',
                                                      'lambda_l1': 'reg_alpha',
                                                      'lambda_l2': 'reg_lambda'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_LGBClassifier_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_LGBClassifier_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_LGBClassifier_holdout_evaluator.txt')  # noqa

        TestHelper.check_plot('data/test_ModelWrappers/test_LightGBMClassifier_plot_feature_importance.png',
                              lambda: fitter.model.plot_feature_importance())

        ######################################################################################################
        # Compare against actual model
        ######################################################################################################
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        pipeline = TransformerPipeline(transformations=transformations)

        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns='Survived')

        holdout_x = data.iloc[holdout_indexes].drop(columns='Survived')

        transformed_training_data = pipeline.fit_transform(training_x)
        transformed_holdout_data = pipeline.transform(holdout_x)

        model = LightGBMClassifier()
        model.train(data_x=transformed_training_data, data_y=training_y, hyper_params=LightGBMHP())
        training_predictions = model.predict(data_x=transformed_training_data)
        holdout_predictions = model.predict(data_x=transformed_holdout_data)

        from lightgbm import LGBMClassifier
        lgbm_model = LGBMClassifier()
        lgbm_model.fit(X=transformed_training_data, y=training_y)
        lgbm_training_predictions = pd.DataFrame(lgbm_model.predict_proba(X=transformed_training_data))
        lgbm_holdout_predictions = pd.DataFrame(lgbm_model.predict_proba(X=transformed_holdout_data))

        assert all(training_predictions[0].values == lgbm_training_predictions[0].values)
        assert all(training_predictions[1].values == lgbm_training_predictions[1].values)

        assert all(holdout_predictions[0].values == lgbm_holdout_predictions[0].values)
        assert all(holdout_predictions[1].values == lgbm_holdout_predictions[1].values)

    def test_LGBClassifier_non_defaults(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        fitter = ModelTrainer(model=LightGBMClassifier(boosting_type=LightGBMBoostingType.RANDOM_FOREST,
                                                       seed=666),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)))
        hp = LightGBMHP(num_leaves=33,
                           min_data_in_leaf=22,
                           max_depth=55,
                           bagging_fraction=0.5,
                           bagging_freq=1,
                           feature_fraction=0.7,
                           lambda_l1=1,
                           lambda_l2=2,
                           learning_rate=0.9,
                           max_bin=265,
                           min_gain_to_split=0.1,
                           min_sum_hessian_in_leaf=0.11,
                           n_estimators=111,
                           # is_unbalanced=False,
                           scale_pos_weight=1.1)
        fitter.train_predict_eval(data=data,
                                  target_variable='Survived',
                                  hyper_params=hp)

        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa

        assert fitter.model.model_object.get_params() == {'boosting_type': 'random_forest',
                                                          'class_weight': None,
                                                          'colsample_bytree': 0.7,
                                                          'importance_type': 'split',
                                                          'learning_rate': 0.9,
                                                          'max_depth': 55,
                                                          'min_child_samples': 22,
                                                          'min_child_weight': 0.11,
                                                          'min_split_gain': 0.1,
                                                          'n_estimators': 111,
                                                          'n_jobs': -1,
                                                          'num_leaves': 33,
                                                          'objective': 'binary',
                                                          'random_state': 666,
                                                          'reg_alpha': 1,
                                                          'reg_lambda': 2,
                                                          'silent': True,
                                                          'subsample': 0.5,
                                                          'subsample_for_bin': 200000,
                                                          'subsample_freq': 1,
                                                          'scale_pos_weight': 1.1,
                                                          'max_bin': 265}

        TestHelper.assert_hyper_params_match(hp, fitter.model,
                                             mapping={'min_gain_to_split': 'min_split_gain',
                                                      'min_sum_hessian_in_leaf': 'min_child_weight',
                                                      'min_data_in_leaf': 'min_child_samples',
                                                      'bagging_fraction': 'subsample',
                                                      'bagging_freq': 'subsample_freq',
                                                      'feature_fraction': 'colsample_bytree',
                                                      'lambda_l1': 'reg_alpha',
                                                      'lambda_l2': 'reg_lambda'})
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_LGBClassifier_non_defaults_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_LGBClassifier_non_defaults_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_LGBClassifier_non_defaults_holdout_evaluator.txt')  # noqa

        TestHelper.check_plot('data/test_ModelWrappers/test_LightGBMClassifier_non_default_plot_feature_importance.png',  # noqa
                              lambda: fitter.model.plot_feature_importance())

    def test_LightGBMRegressor(self):
        data = TestHelper.get_cement_data()
        transformations = None

        fitter = ModelTrainer(model=LightGBMRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        hp = LightGBMHP()
        fitter.train_predict_eval(data=data, target_variable='strength', hyper_params=hp)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.model_object.get_params() == {'boosting_type': 'gbdt',
                                                          'class_weight': None,
                                                          'colsample_bytree': 1.0,
                                                          'importance_type': 'split',
                                                          'learning_rate': 0.1,
                                                          'max_depth': -1,
                                                          'min_child_samples': 20,
                                                          'min_child_weight': 0.001,
                                                          'min_split_gain': 0.0,
                                                          'n_estimators': 100,
                                                          'n_jobs': -1,
                                                          'num_leaves': 31,
                                                          'objective': 'regression',
                                                          'random_state': 42,
                                                          'reg_alpha': 0.0,
                                                          'reg_lambda': 0.0,
                                                          'silent': True,
                                                          'subsample': 1.0,
                                                          'subsample_for_bin': 200000,
                                                          'subsample_freq': 0,
                                                          # 'scale_pos_weight': 1.0,
                                                          'max_bin': 255}

        TestHelper.assert_hyper_params_match(hyper_params=hp, model=fitter.model,
                                             mapping={'min_gain_to_split': 'min_split_gain',
                                                      'min_sum_hessian_in_leaf': 'min_child_weight',
                                                      'min_data_in_leaf': 'min_child_samples',
                                                      'bagging_fraction': 'subsample',
                                                      'bagging_freq': 'subsample_freq',
                                                      'feature_fraction': 'colsample_bytree',
                                                      'lambda_l1': 'reg_alpha',
                                                      'lambda_l2': 'reg_lambda'},
                                             remove_keys=['scale_pos_weight'])
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_LightGBMRegressor_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_LightGBMRegressor_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_LightGBMRegressor_holdout_evaluator.txt')  # noqa

        ######################################################################################################
        # Compare against actual model
        ######################################################################################################
        data = TestHelper.get_cement_data()
        target_variable = 'strength'

        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns=target_variable)

        holdout_x = data.iloc[holdout_indexes].drop(columns=target_variable)

        model = LightGBMRegressor()
        model.train(data_x=training_x, data_y=training_y, hyper_params=LightGBMHP())
        training_predictions = model.predict(data_x=training_x)
        holdout_predictions = model.predict(data_x=holdout_x)

        from lightgbm import LGBMRegressor
        lgbm_model = LGBMRegressor()
        lgbm_model.fit(X=training_x, y=training_y)
        lgbm_training_predictions = lgbm_model.predict(X=training_x)
        lgbm_holdout_predictions = lgbm_model.predict(X=holdout_x)

        assert all(training_predictions == lgbm_training_predictions)
        assert all(training_predictions == lgbm_training_predictions)

        assert all(holdout_predictions == lgbm_holdout_predictions)
        assert all(holdout_predictions == lgbm_holdout_predictions)

    def test_LightGBMRegressor_non_defualt_params(self):
        data = TestHelper.get_cement_data()
        transformations = None

        fitter = ModelTrainer(model=LightGBMRegressor(boosting_type=LightGBMBoostingType.RANDOM_FOREST,
                                                      seed=666),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        hp = LightGBMHP(num_leaves=33, min_data_in_leaf=22, max_depth=55, bagging_fraction=0.5,
                           bagging_freq=1, feature_fraction=0.7, lambda_l1=1, lambda_l2=2, learning_rate=0.9,
                           max_bin=265, min_gain_to_split=0.1, min_sum_hessian_in_leaf=0.11, n_estimators=111)
        fitter.train_predict_eval(data=data,
                                  target_variable='strength',
                                  hyper_params=hp)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa

        assert fitter.model.model_object.get_params() == {'boosting_type': 'random_forest',
                                                          'class_weight': None,
                                                          'colsample_bytree': 0.7,
                                                          'importance_type': 'split',
                                                          'learning_rate': 0.9,
                                                          'max_depth': 55,
                                                          'min_child_samples': 22,
                                                          'min_child_weight': 0.11,
                                                          'min_split_gain': 0.1,
                                                          'n_estimators': 111,
                                                          'n_jobs': -1,
                                                          'num_leaves': 33,
                                                          'objective': 'regression',
                                                          'random_state': 666,
                                                          'reg_alpha': 1,
                                                          'reg_lambda': 2,
                                                          'silent': True,
                                                          'subsample': 0.5,
                                                          'subsample_for_bin': 200000,
                                                          'subsample_freq': 1,
                                                          # 'scale_pos_weight': 1.1,
                                                          'max_bin': 265}

        TestHelper.assert_hyper_params_match(hp, fitter.model,
                                             mapping={'min_gain_to_split': 'min_split_gain',
                                                      'min_sum_hessian_in_leaf': 'min_child_weight',
                                                      'min_data_in_leaf': 'min_child_samples',
                                                      'bagging_fraction': 'subsample',
                                                      'bagging_freq': 'subsample_freq',
                                                      'feature_fraction': 'colsample_bytree',
                                                      'lambda_l1': 'reg_alpha',
                                                      'lambda_l2': 'reg_lambda'},
                                             remove_keys=['scale_pos_weight'])
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_LightGBMRegressor_non_defualt_params_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_LightGBMRegressor_non_defualt_params_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_LightGBMRegressor_non_defualt_params_holdout_evaluator.txt')  # noqa

    def test_XGBoostClassifier(self):
        data = TestHelper.get_titanic_data()
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        fitter = ModelTrainer(model=XGBoostClassifier(),
                              model_transformations=transformations,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                 converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1)))
        hp = XGBoostTreeHP(objective=XGBObjective.BINARY_LOGISTIC)
        fitter.train_predict_eval(data=data, target_variable='Survived', hyper_params=hp)  # noqa
        assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
        assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
        assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
        assert fitter.model.hyper_params.params_dict == {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'verbosity': 0, 'objective': 'binary:logistic', 'booster': 'gbtree', 'n_jobs': 1, 'nthread': 1, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'missing': np.nan}  # noqa

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_XGBoostClassifier_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_XGBoostClassifier_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_XGBoostClassifier_holdout_evaluator.txt')  # noqa

        expected_importance_feature_order = ['Sex_female', 'Pclass_3', 'Pclass_1', 'Embarked_S', 'Age', 'Fare',  # noqa
                                             'Embarked_C', 'Parch_2', 'SibSp_3', 'SibSp_4', 'Parch_0',
                                             'Pclass_2', 'Parch_1', 'SibSp_0', 'SibSp_1', 'Embarked_Q']
        assert all(fitter.model.feature_importance.index.values == expected_importance_feature_order)

        TestHelper.check_plot('data/test_ModelWrappers/test_XGBoostClassifier_plot_feature_importance.png',
                              lambda: fitter.model.plot_feature_importance())

    def test_XGBoostClassifier_early_stopping(self):
        warnings.filterwarnings("ignore")
        # noinspection PyUnusedLocal
        with patch('sys.stdout', new=MockDevice()) as fake_out:  # suppress output of early stopping
            ##################################################################################################
            # early stopping with 100 estimators (default), using training set; stops at 100
            ##################################################################################################
            data = TestHelper.get_titanic_data()
            transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                               CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                               ImputationTransformer(),
                               DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
            fitter = ModelTrainer(model=XGBoostClassifier(early_stopping_rounds=10,
                                                          eval_metric=XGBEvalMetric.AUC),
                                  model_transformations=transformations,
                                  splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                  evaluator=TwoClassProbabilityEvaluator(
                                     converter=TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=1)))
            hp = XGBoostTreeHP(objective=XGBObjective.BINARY_LOGISTIC, n_estimators=100)
            fitter.train_predict_eval(data=data,
                                      target_variable='Survived',
                                      hyper_params=hp)
            assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
            assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
            assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
            assert fitter.model.hyper_params.params_dict == {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'verbosity': 0, 'objective': 'binary:logistic', 'booster': 'gbtree', 'n_jobs': 1, 'nthread': 1, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'missing': np.nan}  # noqa

            TestHelper.assert_hyper_params_match(hp, fitter.model)
            TestHelper.save_string(fitter,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping_string.txt')
            TestHelper.save_string(fitter.training_evaluator,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping_training_evaluator.txt')  # noqa
            TestHelper.save_string(fitter.holdout_evaluator,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping_holdout_evaluator.txt')  # noqa
            ##################################################################################################
            # early stopping with 1000 estimators, using training set
            ##################################################################################################
            transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                               CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                               ImputationTransformer(),
                               DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
            fitter = ModelTrainer(model=XGBoostClassifier(early_stopping_rounds=10,
                                                          eval_metric=XGBEvalMetric.AUC),
                                  model_transformations=transformations,
                                  splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                  evaluator=TwoClassProbabilityEvaluator(
                                     converter=TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=1)))
            hp = XGBoostTreeHP(objective=XGBObjective.BINARY_LOGISTIC, n_estimators=1000)
            fitter.train_predict_eval(data=data,
                                      target_variable='Survived',
                                      hyper_params=hp)
            assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
            assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
            assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
            assert fitter.model.hyper_params.params_dict == {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 1000, 'verbosity': 0, 'objective': 'binary:logistic', 'booster': 'gbtree', 'n_jobs': 1, 'nthread': 1, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'missing': np.nan}  # noqa
            TestHelper.assert_hyper_params_match(hp, fitter.model)
            TestHelper.save_string(fitter,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping2_string.txt')  # noqa
            TestHelper.save_string(fitter.training_evaluator,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping2_training_evaluator.txt')  # noqa
            TestHelper.save_string(fitter.holdout_evaluator,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping2_holdout_evaluator.txt')  # noqa

            ##################################################################################################
            # early stopping with 1000 estimators, using holdout set
            ##################################################################################################
            transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                               CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                               ImputationTransformer(),
                               DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

            _, _, h_x, h_y, _ = OOLearningHelpers.get_final_datasets(data=data,
                                                                    target_variable='Survived',
                                                                    splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),  # noqa
                                                                    transformations=[x.clone() for x in transformations])  # noqa

            fitter = ModelTrainer(model=XGBoostClassifier(early_stopping_rounds=10,
                                                          eval_metric=XGBEvalMetric.AUC,
                                                          eval_set=[(h_x, h_y)]),  # training
                                  model_transformations=transformations,
                                  splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                                  evaluator=TwoClassProbabilityEvaluator(
                                     converter=TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=1)))
            hp = XGBoostTreeHP(objective=XGBObjective.BINARY_LOGISTIC, n_estimators=1000)
            fitter.train_predict_eval(data=data,
                                      target_variable='Survived',
                                      hyper_params=hp)
            assert isinstance(fitter.training_evaluator, TwoClassProbabilityEvaluator)
            assert isinstance(fitter.holdout_evaluator, TwoClassProbabilityEvaluator)
            assert fitter.model.feature_names == ['Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # noqa
            assert fitter.model.hyper_params.params_dict == {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 1000, 'verbosity': 0, 'objective': 'binary:logistic', 'booster': 'gbtree', 'n_jobs': 1, 'nthread': 1, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'missing': np.nan}  # noqa
            TestHelper.assert_hyper_params_match(hp, fitter.model)
            TestHelper.save_string(fitter,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping3_string.txt')  # noqa
            TestHelper.save_string(fitter.training_evaluator,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping3_training_evaluator.txt')  # noqa
            TestHelper.save_string(fitter.holdout_evaluator,
                                   'data/test_ModelWrappers/test_XGBoostClassifier_early_stopping3_holdout_evaluator.txt')  # noqa

    def test_XGBoostClassifier_multiclass(self):
        data = TestHelper.get_iris_data()
        target_variable = 'species'
        fitter = ModelTrainer(model=XGBoostClassifier(),
                              model_transformations=None,
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                              evaluator=MultiClassEvaluator(converter=HighestValueConverter()))
        hp = XGBoostTreeHP(objective=XGBObjective.MULTI_SOFTPROB)
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=hp)
        assert isinstance(fitter.training_evaluator, MultiClassEvaluator)
        assert isinstance(fitter.holdout_evaluator, MultiClassEvaluator)
        assert fitter.model.feature_names == ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        assert fitter.model.hyper_params.params_dict == {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'verbosity': 0, 'objective': 'multi:softprob', 'booster': 'gbtree', 'n_jobs': 1, 'nthread': 1, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'missing': np.nan}  # noqa

        TestHelper.assert_hyper_params_match(hyper_params=hp, model=fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_XGBoostClassifier_multiclass_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_XGBoostClassifier_multiclass_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_XGBoostClassifier_multiclass_holdout_evaluator.txt')  # noqa

        TestHelper.check_plot('data/test_ModelWrappers/test_XGBoostClassifier_plot_feature_importance_multi.png',  # noqa
                              lambda: fitter.model.plot_feature_importance())

    def test_XGBoostRegressor_linear(self):
        data = TestHelper.get_cement_data()
        transformations = None

        fitter = ModelTrainer(model=XGBoostRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        hp = XGBoostLinearHP(objective=XGBObjective.REG_LINEAR)
        fitter.train_predict_eval(data=data, target_variable='strength', hyper_params=hp)

        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa
        assert fitter.model.hyper_params.params_dict == {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'verbosity': 0, 'objective': 'reg:linear', 'booster': 'gblinear', 'n_jobs': 1, 'nthread': 1, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0, 'reg_lambda': 0, 'scale_pos_weight': 1, 'base_score': 0.5, 'missing': np.nan}  # noqa

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_XGBoostRegressor_linear_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_XGBoostRegressor_linear_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_XGBoostRegressor_linear_holdout_evaluator.txt')

    def test_XGBoostRegressor_tree(self):
        data = TestHelper.get_cement_data()
        transformations = None

        fitter = ModelTrainer(model=XGBoostRegressor(),
                              model_transformations=transformations,
                              splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=RegressionEvaluator())
        ######################################################################################################
        # test default hyper-parameters
        ######################################################################################################
        hp = XGBoostTreeHP(objective=XGBObjective.REG_LINEAR)
        fitter.train_predict_eval(data=data, target_variable='strength', hyper_params=hp)
        assert isinstance(fitter.training_evaluator, RegressionEvaluator)
        assert isinstance(fitter.holdout_evaluator, RegressionEvaluator)

        assert fitter.model.feature_names == ['cement', 'slag', 'ash', 'water', 'superplastic', 'coarseagg', 'fineagg', 'age']  # noqa
        assert fitter.model.hyper_params.params_dict == {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 100, 'verbosity': 0, 'objective': 'reg:linear', 'booster': 'gbtree', 'n_jobs': 1, 'nthread': 1, 'gamma': 0, 'min_child_weight': 1, 'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'missing': np.nan}  # noqa

        TestHelper.assert_hyper_params_match(hp, fitter.model)
        TestHelper.save_string(fitter,
                               'data/test_ModelWrappers/test_XGBoostRegressor_tree_string.txt')
        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_XGBoostRegressor_tree_training_evaluator.txt')
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_XGBoostRegressor_tree_holdout_evaluator.txt')

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
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_adaboost.predict(data_x=holdout_x)[1]), 0.7812252964426878)  # noqa

        ######################################################################################################
        # VotingStrategy.SOFT
        ######################################################################################################
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=SoftVotingAggregationStrategy())
        # `train_predict_eval()` does nothing, but make sure it doesn't explode in case it is used in a
        # process that automatically calls `train_predict_eval()
        model_aggregator.train(data_x=train_x, data_y=train_y)

        TestHelper.save_string(model_aggregator,
                               'data/test_ModelWrappers/test_ModelAggregator_string.txt')

        assert isinstance(model_aggregator._base_models[0].model, RandomForestClassifier)
        assert isinstance(model_aggregator._base_models[1].model, CartDecisionTreeClassifier)
        assert isinstance(model_aggregator._base_models[2].model, AdaBoostClassifier)
        assert isinstance(model_aggregator._base_models[0].hyper_params, RandomForestHP)
        assert isinstance(model_aggregator._base_models[1].hyper_params, CartDecisionTreeHP)
        assert isinstance(model_aggregator._base_models[2].hyper_params, AdaBoostClassifierHP)
        assert model_aggregator._base_transformation_pipeline[0].transformations is None
        assert model_aggregator._base_transformation_pipeline[1].transformations is None
        assert model_aggregator._base_transformation_pipeline[2].transformations is None

        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        TestHelper.save_df(voting_predictions, "data/test_ModelWrappers/test_ModelAggregator_soft.txt")

        # make sure we are getting the correct averages back
        assert all([isclose(x, y) for x, y in zip(voting_predictions[0], died_averages)])
        assert all([isclose(x, y) for x, y in zip(voting_predictions[1], survived_averages)])
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_aggregator.predict(data_x=holdout_x)[1]), 0.8075098814229249)  # noqa

        ######################################################################################################
        # VotingStrategy.HARD
        ######################################################################################################
        # for HARD voting, need to pass converters as well
        converter = TwoClassThresholdConverter(positive_class=1)
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=HardVotingAggregationStrategy(converters=[copy.deepcopy(converter) for _ in range(0, 3)]))  # noqa
        # `train_predict_eval()` does nothing, but make sure it doesn't explode in case it is used in a
        # process that automatically calls `train_predict_eval()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        TestHelper.save_df(voting_predictions, "data/test_ModelWrappers/test_ModelAggregator_hard.txt")
        # with open(file, 'wb') as output:
        #     pickle.dump(voting_predictions, output, pickle.HIGHEST_PROTOCOL)

        # [1 if x > 0.5 else 0 for x in predictions_random_forest[1]]
        # [1 if x > 0.5 else 0 for x in model_aggregator.predict(data_x=holdout_x)[1]]
        # list(holdout_y)
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_aggregator.predict(data_x=holdout_x)[1]), 0.7752305665349144)  # noqa

    def test_ModelAggregator_soft_median(self):
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
        died_averages = np.median([predictions_random_forest[0].values, predictions_decision_tree[0].values,
                                   predictions_adaboost[0].values], axis=0)
        survived_averages = np.median([predictions_random_forest[1].values, predictions_decision_tree[1].values,  # noqa
                                       predictions_adaboost[1].values], axis=0)

        assert isclose(roc_auc_score(y_true=holdout_y, y_score=predictions_random_forest[1]), 0.8230566534914362)  # noqa
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_decision_tree.predict(data_x=holdout_x)[1]), 0.772463768115942)  # noqa
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_adaboost.predict(data_x=holdout_x)[1]), 0.7812252964426878)  # noqa

        ######################################################################################################
        # VotingStrategy.SOFT
        ######################################################################################################
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=SoftVotingAggregationStrategy(aggregation=np.median))  # noqa
        # `train_predict_eval()` does nothing, but make sure it doesn't explode in case it is used in a
        # process that automatically calls `train_predict_eval()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        TestHelper.save_string(model_aggregator,
                               'data/test_ModelWrappers/test_ModelAggregator_soft_median_string.txt')
        assert isinstance(model_aggregator._base_models[0].model, RandomForestClassifier)
        assert isinstance(model_aggregator._base_models[1].model, CartDecisionTreeClassifier)
        assert isinstance(model_aggregator._base_models[2].model, AdaBoostClassifier)
        assert isinstance(model_aggregator._base_models[0].hyper_params, RandomForestHP)
        assert isinstance(model_aggregator._base_models[1].hyper_params, CartDecisionTreeHP)
        assert isinstance(model_aggregator._base_models[2].hyper_params, AdaBoostClassifierHP)
        assert model_aggregator._base_transformation_pipeline[0].transformations is None
        assert model_aggregator._base_transformation_pipeline[1].transformations is None
        assert model_aggregator._base_transformation_pipeline[2].transformations is None

        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        # make sure we are getting the correct averages back
        assert all([isclose(x, y) for x, y in zip(voting_predictions[0], died_averages)])
        assert all([isclose(x, y) for x, y in zip(voting_predictions[1], survived_averages)])
        assert isclose(roc_auc_score(y_true=holdout_y, y_score=model_aggregator.predict(data_x=holdout_x)[1]), 0.800395256916996)  # noqa

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
        # `train_predict_eval()` does nothing, but make sure it doesn't explode in case it is used in a
        # process that automatically calls `train_predict_eval()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        TestHelper.save_string(model_aggregator,
                               'data/test_ModelWrappers/test_ModelAggregator_multi_class_string.txt')
        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        # make sure we are getting the correct averages back
        assert all([isclose(x, y) for x, y in zip(voting_predictions['setosa'], setosa_averages)])
        assert all([isclose(x, y) for x, y in zip(voting_predictions['versicolor'], versicolor_averages)])
        assert all([isclose(x, y) for x, y in zip(voting_predictions['virginica'], virginica_averages)])

        TestHelper.save_df(voting_predictions, "data/test_ModelWrappers/test_ModelAggregator_multi_class_soft_predictions.txt")  # noqa

        evaluator.evaluate(actual_values=holdout_y, predicted_values=voting_predictions)
        TestHelper.save_string(evaluator,
                               "data/test_ModelWrappers/test_ModelAggregator_multi_class_soft_evaluator.txt")  # noqa

        ######################################################################################################
        # VotingStrategy.HARD
        ######################################################################################################
        # for HARD voting, need to pass converters as well
        converter = HighestValueConverter()
        model_infos = [ModelInfo(model=RandomForestClassifier(), hyper_params=RandomForestHP()),
                       ModelInfo(model=CartDecisionTreeClassifier(), hyper_params=CartDecisionTreeHP()),
                       ModelInfo(model=AdaBoostClassifier(), hyper_params=AdaBoostClassifierHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=HardVotingAggregationStrategy(converters=[copy.deepcopy(converter) for _ in range(0, 3)]))  # noqa
        # `train_predict_eval()` does nothing, but make sure it doesn't explode in case it is used in a
        # process that automatically calls `train_predict_eval()
        model_aggregator.train(data_x=train_x, data_y=train_y)
        TestHelper.save_string(model_aggregator,
                               'data/test_ModelWrappers/test_ModelAggregator_multi_class_hard_string.txt')
        voting_predictions = model_aggregator.predict(data_x=holdout_x)

        assert all(voting_predictions.index.values == holdout_x.index.values)
        assert all(voting_predictions.columns.values == predictions_random_forest.columns.values)

        # predictions should all sum to 1
        assert all([sum(voting_predictions.loc[x]) == 1.0 for x in voting_predictions.index.values])

        TestHelper.save_df(voting_predictions,
                           "data/test_ModelWrappers/test_ModelAggregator_multi_class_hard_predictions.txt")  # noqa

        evaluator.evaluate(actual_values=holdout_y, predicted_values=voting_predictions)
        TestHelper.save_string(evaluator,
                               "data/test_ModelWrappers/test_ModelAggregator_multi_class_hard_evaluator.txt")  # noqa

    def test_ModelAggregator_regression(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_regression(data, target_variable)  # noqa

        model_linear_regression = LinearRegressorSK()
        model_linear_regression.train(data_x=train_x, data_y=train_y)
        predictions_linear_regression = model_linear_regression.predict(data_x=holdout_x)

        model_cart = CartDecisionTreeRegressor()
        model_cart.train(data_x=train_x, data_y=train_y, hyper_params=CartDecisionTreeHP(criterion='mse'))
        predictions_cart = model_cart.predict(data_x=holdout_x)

        model_adaboost = AdaBoostRegressor()
        model_adaboost.train(data_x=train_x, data_y=train_y, hyper_params=AdaBoostRegressorHP())
        predictions_adaboost = model_adaboost.predict(data_x=holdout_x)

        expected_aggregation = (predictions_linear_regression + predictions_cart + predictions_adaboost) / 3

        model_infos = [ModelInfo(model=LinearRegressorSK(), hyper_params=None),
                       ModelInfo(model=CartDecisionTreeRegressor(), hyper_params=CartDecisionTreeHP(criterion='mse')),  # noqa
                       ModelInfo(model=AdaBoostRegressor(), hyper_params=AdaBoostRegressorHP())]
        model_aggregator = ModelAggregator(base_models=model_infos,
                                           aggregation_strategy=MeanAggregationStrategy())
        model_aggregator.train(data_x=train_x, data_y=train_y)
        TestHelper.save_string(model_aggregator,
                               'data/test_ModelWrappers/test_ModelAggregator_regression_string.txt')

        predictions_aggregation = model_aggregator.predict(data_x=holdout_x)

        assert all([isclose(x, y) for x, y in zip(expected_aggregation, predictions_aggregation)])

        TestHelper.save_df(pd.Series(predictions_aggregation),
                           'data/test_ModelWrappers/test_ModelAggregator_regression_predictions.csv')

    def test_ModelAggregator_regression_transformations(self):
        """
        Make sure transformations work as expected for each
        """
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_regression(data,
                                                                                           target_variable)
        model_infos = [ModelInfo(model=LinearRegressorSK(),
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
                                           aggregation_strategy=MeanAggregationStrategy(),
                                           parallelization_cores=-1)
        model_aggregator.train(data_x=train_x, data_y=train_y)

        # check that all the models and transformations are the types (in the order) expected
        assert isinstance(model_aggregator._base_models[0].model, LinearRegressorSK)
        assert isinstance(model_aggregator._base_models[1].model, CartDecisionTreeRegressor)
        assert isinstance(model_aggregator._base_models[2].model, AdaBoostRegressor)
        assert model_aggregator._base_models[0].hyper_params is None
        assert isinstance(model_aggregator._base_models[1].hyper_params, CartDecisionTreeHP)
        assert isinstance(model_aggregator._base_models[2].hyper_params, AdaBoostRegressorHP)
        assert isinstance(model_aggregator._base_transformation_pipeline[0].transformations[0], PolynomialFeaturesTransformer)  # noqa
        assert isinstance(model_aggregator._base_transformation_pipeline[0].transformations[1], DummyEncodeTransformer)  # noqa
        assert isinstance(model_aggregator._base_transformation_pipeline[1].transformations[0], CenterScaleTransformer)  # noqa
        assert isinstance(model_aggregator._base_transformation_pipeline[1].transformations[1], DummyEncodeTransformer)  # noqa
        assert isinstance(model_aggregator._base_transformation_pipeline[2].transformations[0], DummyEncodeTransformer)  # noqa

        # check polynomial fields
        TestHelper.save_df(model_aggregator._base_models[0].model.data_x_trained_head,
                           'data/test_ModelWrappers/test_ModelAggregator_LinearRegressor.csv')

        TestHelper.save_df(model_aggregator._base_models[1].model.data_x_trained_head,
                           'data/test_ModelWrappers/test_ModelAggregator_CartDecisionTreeRegressor.csv')

        TestHelper.save_df(model_aggregator._base_models[2].model.data_x_trained_head,
                           'data/test_ModelWrappers/test_ModelAggregator_AdaBoostRegressor.csv')

        predictions_aggregation = model_aggregator.predict(data_x=holdout_x)
        evaluator = RegressionEvaluator()
        evaluator.evaluate(actual_values=holdout_y, predicted_values=predictions_aggregation)
        TestHelper.save_string(evaluator,
                               'data/test_ModelWrappers/test_ModelAggregator_regression_transformations_evaluator.csv')  # noqa

    def test_ModelAggregator_regression_median_transformations(self):
        """
        Make sure transformations work as expected for each
        """
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'
        train_x, train_y, holdout_x, holdout_y = TestHelper.split_train_holdout_regression(data,
                                                                                           target_variable)
        model_infos = [ModelInfo(model=LinearRegressorSK(),
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

        TestHelper.save_df(model_aggregator._base_models[0].model.data_x_trained_head,
                           'data/test_ModelWrappers/test_ModelAggregator_regression_median_transformations_LinearRegressor.csv')  # noqa

        TestHelper.save_df(model_aggregator._base_models[1].model.data_x_trained_head,
                           'data/test_ModelWrappers/test_ModelAggregator_regression_median_transformations_CartDecisionTreeRegressor.csv')  # noqa

        TestHelper.save_df(model_aggregator._base_models[2].model.data_x_trained_head,
                           'data/test_ModelWrappers/test_ModelAggregator_regression_median_transformations_AdaBoostRegressor.csv')  # noqa

        predictions_aggregation = model_aggregator.predict(data_x=holdout_x)
        evaluator = RegressionEvaluator()
        evaluator.evaluate(actual_values=holdout_y, predicted_values=predictions_aggregation)
        TestHelper.save_string(evaluator,
                               'data/test_ModelWrappers/test_ModelAggregator_regression_median_transformations_evaluator.csv')  # noqa

    def helper_test_ModelStacker_Classification(self, data, positive_class, cache_directory=None):
        """
        NOTE: WHEN USING cache_directory THIS FUNCTION DOESN'T CHECK WHETHER OR NOT THE CACHED FILES EXIST.
            SO YOU CAN TEST AS IF RUNNING THE SEARCHER FOR "FIRST TIME", OR "SUBSEQUENT RUNS" (i.e. cache
            files already exist)
        """
        target_variable = 'Survived'
        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class)),  # noqa
                      SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class))]  # noqa

        def fh(file):
            # file helper
            return TestHelper.ensure_test_directory(os.path.join('data/test_ModelWrappers/helper_test_ModelStacker_Classification',  # noqa
                                                                 file))

        cart_base_model = ModelInfo(description='cart',
                                    model=CartDecisionTreeClassifier(),
                                    transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                    hyper_params=CartDecisionTreeHP())
        rf_base_model = ModelInfo(description='random_forest',
                                  model=RandomForestClassifier(),
                                  transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                  hyper_params=RandomForestHP())
        base_models = [cart_base_model, rf_base_model]

        # Test model stacker with duplicate model names; should get assertion error
        self.assertRaises(AssertionError,
                          lambda: ModelStacker(base_models=base_models + [cart_base_model],
                                               scores=score_list,
                                               stacking_model=LogisticClassifier(),
                                               converter=ExtractPredictionsColumnConverter(column=positive_class)))  # noqa

        # Use same splitter information to get the training/holdout data
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)
        expected_train_y = data.iloc[training_indexes].Survived
        expected_train_x = data.iloc[training_indexes].drop(columns=target_variable)
        expected_test_x = data.iloc[test_indexes].drop(columns=target_variable)

        file_train_callback = fh('train_callback_train_meta.pkl')
        file_test_callback_traindata = fh('test_callback_test_meta_traindata.pkl')
        file_test_callback_holdoutdata = fh('test_callback_test_meta_holdoutdata.pkl')

        # these variables help us verify the callbacks were called
        train_callback_called = list()
        predict_callback_called = list()

        def train_callback(train_meta_x, train_meta_y, hyper_params):
            # the `train_meta_x` that we built in `train_predict_eval()` should have the same indexes as
            # `train_meta_x
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
                                     converter=ExtractPredictionsColumnConverter(column=positive_class),
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
            fitter.train_predict_eval(data=data,
                                      target_variable=target_variable,
                                      hyper_params=LogisticClassifierHP())
            time_stop = time.time()
            fit_time = time_stop - time_start

            assert os.path.isdir(fitter._persistence_manager._cache_directory)
            expected_stacker_cached_file = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification/ModelStacker_LogisticClassifier_penalty_l2_regularization_inverse_1.0_solver_liblinear.pkl')  # noqa
            expected_base_cart_cached_file = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification/base_cart_criterion_gini_splitter_best_max_depth_None_min_samples_split_2_min_samples_leaf_1_min_weight_fraction_leaf_0.0_max_leaf_nodes_None_max_features_None.pkl')  # noqa
            expected_base_rf_cached_file = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification/base_random_forest_n_estimators_500_criterion_gini_max_features_None_max_depth_None_min_samples_split_2_min_samples_leaf_1_min_weight_fraction_leaf_0.0_max_leaf_nodes_None_min_impurity_decrease_0.0_bootstrap_True_oob_score_False.pkl')  # noqa
            expected_train_meta_file = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification/train_meta.pkl')  # noqa
            # ensure the cache path of the stacker is the final stacked model
            assert fitter._persistence_manager._cache_path == expected_stacker_cached_file
            assert os.path.isfile(fitter._persistence_manager._cache_path)
            # each base model should have the following file,
            # corresponding to the final trained model for each base model
            assert os.path.isfile(expected_base_cart_cached_file)
            assert os.path.isfile(expected_base_rf_cached_file)
            # make sure the `train_meta` dataset is cached
            assert os.path.isfile(expected_train_meta_file)

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
            fitter.train_predict_eval(data=data,
                                      target_variable=target_variable,
                                      hyper_params=LogisticClassifierHP())
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
        file_test_model_stacker_resample_data_cart = fh('resample_data_cart.pkl')
        file_test_model_stacker_resample_data_rf = fh('resample_data_rf.pkl')
        file_test_model_stacker_resample_means = fh('resample_means.pkl')
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_cart,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='cart'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_rf,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='random_forest'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_means,
                                                     expected_dataframe=fitter.model.get_resample_means())

        if OOLearningHelpers.is_series_numeric(data[target_variable]):
            file_test_model_stacker_train_meta_correlations = fh('train_meta_correlations.pkl')
            file_plot_correlations = 'data/test_ModelWrappers/helper_test_ModelStacker_Classification/stacker_correlations.png'  # noqa
        else:
            file_test_model_stacker_train_meta_correlations = fh('train_meta_correlations_string_target.pkl')
            file_plot_correlations = 'data/test_ModelWrappers/helper_test_ModelStacker_Classification/stacker_correlations_string_target.png'  # noqa

        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_train_meta_correlations,
                                                     expected_dataframe=fitter.model._train_meta_correlations)

        TestHelper.check_plot(file_plot_correlations, lambda: fitter.model.plot_correlation_heatmap(),
                              set_size_w_h=None)

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

    def test_ModelStacker_Classification_no_converter(self):
        """
        NOTE: when using base-models that return DataFrame's for `predict()`, we need to supply a converter
        """
        data = TestHelper.get_titanic_data()
        positive_class = 1
        target_variable = 'Survived'
        score_list = [KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class)),  # noqa
                      SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class))]  # noqa

        cart_base_model = ModelInfo(description='cart',
                                    model=CartDecisionTreeClassifier(),
                                    transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                    hyper_params=CartDecisionTreeHP())
        rf_base_model = ModelInfo(description='random_forest',
                                  model=RandomForestClassifier(),
                                  transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                  hyper_params=RandomForestHP())
        base_models = [cart_base_model, rf_base_model]

        model_stacker = ModelStacker(base_models=base_models,
                                     scores=score_list,
                                     stacking_model=LogisticClassifier(),
                                     stacking_transformations=None,
                                     # no converter, so we should get an Assertion error
                                     # converter=ExtractPredictionsColumnConverter(column=positive_class),
                                     )

        fitter = ModelTrainer(model=model_stacker,
                              model_transformations=[
                                  RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                                  CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                                  ImputationTransformer()],
                              splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                              evaluator=TwoClassProbabilityEvaluator(
                                  converter=TwoClassThresholdConverter(threshold=0.5,
                                                                       positive_class=positive_class)))
        self.assertRaises(AssertionError,
                          lambda: fitter.train_predict_eval(data=data,
                                                            target_variable=target_variable,
                                                            hyper_params=LogisticClassifierHP()))

    def test_ModelStacker_include_original_dataset_in_stacker(self):

        def fhh(file):
            # file helper
            return TestHelper.ensure_test_directory(
                os.path.join('data/test_ModelWrappers/helper_test_ModelStacker_Classification',
                             file))

        def fh(file):
            # file helper
            return TestHelper.ensure_test_directory(
                os.path.join('data/test_ModelWrappers/test_ModelStacker_include_original_dataset_in_stacker',
                             file))

        data = TestHelper.get_titanic_data()
        positive_class = 1
        target_variable = 'Survived'
        score_list = [
            KappaScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class)),
            SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=positive_class))  # noqa
        ]

        global_transformations = [
            RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),  # noqa
            CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),  # noqa
            ImputationTransformer()
        ]
        stacker_transformations = [
            CenterScaleTransformer(),
            DummyEncodeTransformer(),
        ]

        cart_base_model = ModelInfo(description='cart',
                                    model=CartDecisionTreeClassifier(),
                                    transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                    hyper_params=CartDecisionTreeHP())
        rf_base_model = ModelInfo(description='random_forest',
                                  model=RandomForestClassifier(),
                                  transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                  hyper_params=RandomForestHP())
        base_models = [cart_base_model, rf_base_model]

        # Use same splitter information to get the same training/holdout data that is used in the ModelTrainer
        # apply the same transformations
        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, test_indexes = splitter.split(target_values=data.Survived)
        expected_train_y = data.iloc[training_indexes].Survived
        expected_train_x = data.iloc[training_indexes].drop(columns=target_variable)
        all_trans_copy = [x.clone() for x in global_transformations] + \
                         [x.clone() for x in stacker_transformations]
        # this is the dataset that the final base model should be training on (excluding the predictions meta)
        all_trans_pipeline = TransformerPipeline(transformations=all_trans_copy)
        expected_train_x = all_trans_pipeline.fit_transform(expected_train_x)
        expected_test_x = all_trans_pipeline.transform(data.iloc[test_indexes].drop(columns=target_variable))

        # these variables help us verify the callbacks were called
        train_callback_called = list()
        predict_callback_called = list()

        # these are the cached predictions (i.e. meta features)
        file_train_callback = fhh('train_callback_train_meta.pkl')
        file_test_callback_traindata = fhh('test_callback_test_meta_traindata.pkl')
        file_test_callback_holdoutdata = fhh('test_callback_test_meta_holdoutdata.pkl')

        with open(file_train_callback, 'rb') as saved_object:
            expected_predictions_raw = pickle.load(saved_object)

        meta_data_pipeline = TransformerPipeline(transformations=[x.clone() for x in stacker_transformations])
        # predictions from base models that the stacker will use as features
        expected_meta_built = meta_data_pipeline.fit_transform(expected_predictions_raw)

        with open(file_test_callback_traindata, 'rb') as saved_object:
            expected_predictions_raw = pickle.load(saved_object)
        # predictions after all models are trained on entire dataset
        expected_meta_train = meta_data_pipeline.transform(expected_predictions_raw)

        with open(file_test_callback_holdoutdata, 'rb') as saved_object:
            expected_predictions_raw = pickle.load(saved_object)
        # predictions of base models from holdout data
        expected_meta_test = meta_data_pipeline.transform(expected_predictions_raw)

        def train_callback(train_meta_x, train_meta_y, hyper_params):
            # the `train_meta_x` that we built in `train_predict_eval()` should have the same indexes as
            # `train_meta_x
            assert all(train_meta_x.index.values == expected_train_x.index.values)
            # cached dataframe in `file_train_callback` are the predictions from the base model we expect

            # gets the predictions used from another unit test (where we don't have any stacker
            # tarnsformations and makes sure that once we apply the same transformations, we get the
            # expeted predictions (i.e. features that the stacker will use)
            TestHelper.ensure_all_values_equal(data_frame1=expected_meta_built,
                                               data_frame2=train_meta_x[['cart', 'random_forest']])

            # in addition to using the predictions as features, we are also using the original dataset, but
            # with the global and stacker transformations
            TestHelper.ensure_all_values_equal(data_frame1=expected_train_x,
                                               data_frame2=train_meta_x.drop(columns=['cart', 'random_forest']))  # noqa

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

                TestHelper.ensure_all_values_equal(data_frame1=expected_meta_train,
                                                   data_frame2=test_meta[['cart', 'random_forest']])
                # in addition to using the predictions as features, we are also using the original dataset,
                # but with the global and stacker transformations
                TestHelper.ensure_all_values_equal(data_frame1=expected_train_x,
                                                   data_frame2=test_meta.drop(columns=['cart', 'random_forest']))  # noqa
                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_train')

            elif len(predict_callback_called) == 1:  # second time called i.e. holdout evaluation
                # the `test_meta` that we built in `predict()` should have the same indexes as
                # `expected_test_x` when evaluating the holdout set
                assert all(test_meta.index.values == expected_test_x.index.values)
                # cached dataframe in `file_test_callback_holdoutdata` are the predictions from the base
                # models (refitted on the entire training set) on the dataset that was passed into `predict()`
                # in this case the holdout data, which will then be predicted on the final stacking model
                TestHelper.ensure_all_values_equal(data_frame1=expected_meta_test,
                                                   data_frame2=test_meta[['cart', 'random_forest']])
                # in addition to using the predictions as features, we are also using the original dataset,
                # but with the global and stacker transformations
                TestHelper.ensure_all_values_equal(data_frame1=expected_test_x,
                                                   data_frame2=test_meta.drop(columns=['cart', 'random_forest']))  # noqa

                # add value to the list so we know this callback (and this if) was called
                predict_callback_called.append('predict_called_holdout')
            else:
                raise ValueError()

        model_stacker = ModelStacker(base_models=base_models,
                                     scores=score_list,
                                     stacking_model=LogisticClassifier(),
                                     # transformers will be applied to original dataset and predictions
                                     stacking_transformations=[x.clone() for x in stacker_transformations],
                                     include_original_dataset=True,  # new field
                                     converter=ExtractPredictionsColumnConverter(column=positive_class),
                                     train_callback=train_callback,
                                     predict_callback=predict_callback)

        trainer = ModelTrainer(model=model_stacker,
                               # transformations for all models, not just stackers
                               model_transformations=[x.clone() for x in global_transformations],
                               splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.2),
                               evaluator=TwoClassProbabilityEvaluator(
                                  converter=TwoClassThresholdConverter(threshold=0.5,
                                                                       positive_class=positive_class)))
        trainer.train_predict_eval(data=data,
                                   target_variable=target_variable,
                                   hyper_params=LogisticClassifierHP())

        # verify our callback is called. If it wasn't, we would never know and the assertions wouldn't run.
        assert train_callback_called == ['train_called']
        # `predict_callback` should be called TWICE (once for training eval & once for holdout eval)
        assert predict_callback_called == ['predict_called_train', 'predict_called_holdout']

        actual_metrics = trainer.training_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.9796156766544011, 'AUC Precision/Recall': 0.9716221234154416, 'Kappa': 0.8451830755325135, 'F1 Score': 0.9009708737864077, 'Two-Class Accuracy': 0.9283707865168539, 'Error Rate': 0.07162921348314606, 'True Positive Rate': 0.8498168498168498, 'True Negative Rate': 0.9772209567198178, 'False Positive Rate': 0.022779043280182234, 'False Negative Rate': 0.15018315018315018, 'Positive Predictive Value': 0.9586776859504132, 'Negative Predictive Value': 0.9127659574468086, 'Prevalence': 0.38342696629213485, 'No Information Rate': 0.6165730337078652, 'Total Observations': 712}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])
        # holdout AUC/ROC for CART was: 0.7711462450592885; for Random Forest was 0.8198945981554676;
        # slight increase to 0.8203557312252964; (default hyper-params for all models)
        actual_metrics = trainer.holdout_evaluator.all_quality_metrics
        expected_metrics = {'AUC ROC': 0.8491436100131751, 'AUC Precision/Recall': 0.8231511029698481, 'Kappa': 0.6364251861882192, 'F1 Score': 0.7656250000000001, 'Two-Class Accuracy': 0.8324022346368715, 'Error Rate': 0.16759776536312848, 'True Positive Rate': 0.7101449275362319, 'True Negative Rate': 0.9090909090909091, 'False Positive Rate': 0.09090909090909091, 'False Negative Rate': 0.2898550724637681, 'Positive Predictive Value': 0.8305084745762712, 'Negative Predictive Value': 0.8333333333333334, 'Prevalence': 0.3854748603351955, 'No Information Rate': 0.6145251396648045, 'Total Observations': 179}  # noqa
        assert all([x == y for x, y in zip(actual_metrics.keys(), expected_metrics.keys())])
        assert all([isclose(x, y) for x, y in zip(actual_metrics.values(), expected_metrics.values())])

        # test resample data
        # using ffh which points to the directory `helper_test_ModelStacker_Classification`
        # should have the same resample data since the base transformations and models are the same
        # we are only changing the transformations and training data for the Stacker
        file_test_model_stacker_resample_data_cart = fhh('resample_data_cart.pkl')
        file_test_model_stacker_resample_data_rf = fhh('resample_data_rf.pkl')
        file_test_model_stacker_resample_means = fhh('resample_means.pkl')
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_cart,
                                                     expected_dataframe=trainer.model.get_resample_data(model_description='cart'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_rf,
                                                     expected_dataframe=trainer.model.get_resample_data(model_description='random_forest'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_means,
                                                     expected_dataframe=trainer.model.get_resample_means())

        file_test_model_stacker_train_meta_correlations = fh('train_meta_correlations.pkl')
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_train_meta_correlations,
                                                     expected_dataframe=trainer.model._train_meta_correlations)  # noqa

    def test_ModelStacker_Regression_no_stacker_transformations(self):
        ######################################################################################################
        # elastic_net stacker, svm & GBM & polynomial linear regression base models
        ######################################################################################################
        def fh(file):
            # file helper
            return TestHelper.ensure_test_directory(
                os.path.join('data/test_ModelWrappers/test_ModelStacker_Regression_no_stacker_transformations',  # noqa
                             file))

        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'

        # Use same splitter information to get the training/holdout data
        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, test_indexes = splitter.split(target_values=data[target_variable])
        expected_train_y = data.iloc[training_indexes][target_variable]
        expected_train_x = data.iloc[training_indexes].drop(columns=target_variable)
        expected_test_x = data.iloc[test_indexes].drop(columns=target_variable)

        file_train_callback = fh('train_callback_train_meta.pkl')
        file_test_callback_traindata = fh('test_callback_test_meta_traindata.pkl')
        file_test_callback_holdoutdata = fh('test_callback_test_meta_holdoutdata.pkl')

        # these variables help us verify the callbacks were called
        train_callback_called = list()
        predict_callback_called = list()

        def train_callback(train_meta_x, train_meta_y, hyper_params):
            # the `train_meta_x` that we built in `train_predict_eval()` should have the same indexes as
            # `train_meta_x
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
                                                  self.get_CartDecisionTreeRegressor(),
                                                  self.get_GradientBoostingRegressor()],
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
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=ElasticNetRegressorHP())
        # verify our callback is called. If it wasn't, we would never know and the assertions wouldn't run.
        assert train_callback_called == ['train_called']
        # `predict_callback` should be called TWICE (once for training eval & once for holdout eval)
        assert predict_callback_called == ['predict_called_train', 'predict_called_holdout']

        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_ModelStacker_Regression_no_stacker_transformations_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_ModelStacker_Regression_no_stacker_transformations_holdout_evaluator.txt')  # noqa

        assert [x.description for x in model_stacker._base_models] == ['LinearRegressor_polynomial_2', 'CartDecisionTreeRegressor', 'GradientBoostingRegressor']  # noqa
        # test resample data
        file_test_model_stacker_resample_data_regression = fh('resample_data_regression.pkl')
        file_test_model_stacker_resample_data_cart = fh('resample_data_cart_regression.pkl')
        file_test_model_stacker_resample_data_gb = fh('resample_data_gb.pkl')
        file_test_model_stacker_resample_means = fh('resample_means_regression.pkl')
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_regression,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='LinearRegressor_polynomial_2'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_cart,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='CartDecisionTreeRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_gb,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='GradientBoostingRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_means,
                                                     expected_dataframe=fitter.model.get_resample_means())

        file_test_model_stacker_train_meta_correlations = fh('train_meta_correlations_regression.pkl')
        file_plot_correlations = 'data/test_ModelWrappers/test_ModelStacker_Regression_no_stacker_transformations/stacker_correlations_regression.png'  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_train_meta_correlations,
                                                     expected_dataframe=fitter.model._train_meta_correlations)
        TestHelper.check_plot(file_plot_correlations, lambda: fitter.model.plot_correlation_heatmap(),
                              set_size_w_h=None)

    def test_ModelStacker_Regression_with_stacker_transformations(self):
        ######################################################################################################
        # elastic_net stacker, svm & GBM & polynomial linear regression base models
        ######################################################################################################
        def fh(file):
            # file helper
            return TestHelper.ensure_test_directory(
                os.path.join('data/test_ModelWrappers/test_ModelStacker_Regression_with_stacker_transformations',  # noqa
                             file))
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'

        # Use same splitter information to get the training/holdout data
        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, test_indexes = splitter.split(target_values=data[target_variable])
        expected_train_y = data.iloc[training_indexes][target_variable]
        expected_train_x = data.iloc[training_indexes].drop(columns=target_variable)
        expected_test_x = data.iloc[test_indexes].drop(columns=target_variable)

        file_train_callback = fh('train_callback_train_meta_stacker_trans.pkl')
        file_test_callback_traindata = fh('test_callback_test_meta_traindata_stacker_trans.pkl')
        file_test_callback_holdoutdata = fh('test_callback_test_meta_holdoutdata_stacker_trans.pkl')

        # these variables help us verify the callbacks were called
        train_callback_called = list()
        predict_callback_called = list()

        def train_callback(train_meta_x, train_meta_y, hyper_params):
            # the `train_meta_x` that we built in `train_predict_eval()` should have the same indexes as
            # `train_meta_x
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

        cart_info = self.get_CartDecisionTreeRegressor()
        gbm_info = self.get_GradientBoostingRegressor()

        model_stacker = ModelStacker(base_models=[ModelDefaults.get_LinearRegressor(degrees=2),
                                                  cart_info,
                                                  gbm_info],
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
        fitter.train_predict_eval(data=data,
                                  target_variable=target_variable,
                                  hyper_params=ElasticNetRegressorHP())
        # verify our callback is called. If it wasn't, we would never know and the assertions wouldn't run.
        assert train_callback_called == ['train_called']
        # `predict_callback` should be called TWICE (once for training eval & once for holdout eval)
        assert predict_callback_called == ['predict_called_train', 'predict_called_holdout']

        TestHelper.save_string(fitter.training_evaluator,
                               'data/test_ModelWrappers/test_ModelStacker_Regression_with_stacker_transformations_training_evaluator.txt')  # noqa
        TestHelper.save_string(fitter.holdout_evaluator,
                               'data/test_ModelWrappers/test_ModelStacker_Regression_with_stacker_transformations_holdout_evaluator.txt')  # noqa

        assert all([x == y for x, y in zip([x.description for x in model_stacker._base_models], ['LinearRegressor_polynomial_2', 'CartDecisionTreeRegressor', 'GradientBoostingRegressor'])])  # noqa
        # test resample data
        file_test_model_stacker_resample_data_regression = fh('resample_data_regression_stacker_trans.pkl')
        file_test_model_stacker_resample_data_cart = fh('resample_data_cart_regression_stacker_trans.pkl')
        file_test_model_stacker_resample_data_gb = fh('resample_data_gb_stacker_trans.pkl')
        file_test_model_stacker_resample_means = fh('resample_means_regression_stacker_trans.pkl')
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_regression,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='LinearRegressor_polynomial_2'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_cart,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='CartDecisionTreeRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_data_gb,
                                                     expected_dataframe=fitter.model.get_resample_data(model_description='GradientBoostingRegressor'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_resample_means,
                                                     expected_dataframe=fitter.model.get_resample_means())

        file_test_model_stacker_train_meta_correlations = fh('train_meta_correlations_regression_stacker_trans.pkl')  # noqa
        file_plot_correlations = 'data/test_ModelWrappers/correlations_regression_stacker_trans.png'
        TestHelper.ensure_all_values_equal_from_file(file=file_test_model_stacker_train_meta_correlations,
                                                     expected_dataframe=fitter.model._train_meta_correlations)
        TestHelper.check_plot(file_plot_correlations, lambda: fitter.model.plot_correlation_heatmap())

    # noinspection PyUnusedLocal
    def test_ModelStacker_caching(self):
        cache_directory = TestHelper.ensure_test_directory('data/test_ModelWrappers/cached_test_models/test_ModelStacker_classification')  # noqa

        if os.path.isdir(cache_directory):
            shutil.rmtree(cache_directory)
        assert not os.path.isdir(cache_directory)

        # cache files do not exist
        fit_time_not_previously_cached = self.helper_test_ModelStacker_Classification(data=TestHelper.get_titanic_data(),  # noqa
                                                                                      positive_class=1,
                                                                                      cache_directory=cache_directory)  # noqa
        # print(fit_time_not_previously_cached)
        fit_time_previously_cached = self.helper_test_ModelStacker_Classification(data=TestHelper.get_titanic_data(),  # noqa
                                                                                  positive_class=1,
                                                                                  cache_directory=cache_directory)  # noqa
        # print(fit_time_previously_cached)
        shutil.rmtree(cache_directory)
        # assert 6 < fit_time_not_previously_cached < 8  # looks like around ~7 seconds on average
        # assert fit_time_previously_cached < 2  # improves to less than 2 with caching

    def test_ModelStacker_Tuning(self):
        """
        Ensure that we can re-use all of the objects passed into the ModelStacker so, e.g., we can Tune
        Ensure we are caching correctly.
        """

        ######################################################################################################
        # first, ensure the Lock is truly a singleton
        # internally, the stacker uses a Lock, because building the "meta training" dataset will reuse
        # the same training set built by the stacker. i.e. regardless of which stacking model and/or
        # model hyper-parameters we use, the "meta training" dataset is built exactly the same way, and
        # will also have exactly the same cache file name. This creates problems for multi-threading
        # because multiple threads are reading/writing to the same file. Therefore, we lock the process
        # within ModelStacker._train(). This actually has the side affect of making performance similar to
        # single-threading the first time the "meta training" dataset is built up, but should drastically
        # increase once it is built and cached
        ######################################################################################################
        test1 = ModelStacker(base_models=[], scores=None, stacking_model=None)
        test2 = ModelStacker(base_models=[], scores=None, stacking_model=None)
        assert test1 is not test2  # different objections
        assert test1.StackerMetaLock() is test2.StackerMetaLock()  # same objects via Singleton
        assert test1.StackerBaseLock() is test2.StackerBaseLock()  # same objects via Singleton

        ######################################################################################################
        # now test tuner
        ######################################################################################################
        cache_directory = TestHelper.ensure_test_directory('data/test_ModelWrappers/test_tuning_stacker_caching')  # noqa
        data = TestHelper.get_titanic_data()
        positive_class = 1
        target_variable = 'Survived'

        regularization_inverse = [0.001, 0.01, 0.05, 0.1, 1, 5, 8, 10]
        grid = HyperParamsGrid(params_dict=dict(regularization_inverse=regularization_inverse))

        # noinspection PyShadowingNames
        def create_tuner():

            score_list = [
                KappaScore(converter=TwoClassThresholdConverter(threshold=0.5,
                                                                positive_class=positive_class)),
                SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=positive_class)),
            ]

            cart_base_model = ModelInfo(description='cart',
                                        model=CartDecisionTreeClassifier(),
                                        transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                        hyper_params=CartDecisionTreeHP())
            rf_base_model = ModelInfo(description='random_forest',
                                      model=RandomForestClassifier(),
                                      transformations=[DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)],
                                      hyper_params=RandomForestHP())
            base_models = [cart_base_model, rf_base_model]

            model_stacker = ModelStacker(base_models=base_models,
                                         scores=score_list,
                                         stacking_model=LogisticClassifier(),
                                         stacking_transformations=None,
                                         converter=ExtractPredictionsColumnConverter(column=positive_class))

            transformations = [
                RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                ImputationTransformer()
            ]
            resampler = RepeatedCrossValidationResampler(model=model_stacker,
                                                         transformations=transformations,
                                                         scores=score_list,
                                                         folds=3,
                                                         repeats=1,
                                                         # fold_decorators=[TwoClassThresholdDecorator(parallelization_cores=0)]
                                                         )

            tuner = GridSearchModelTuner(resampler=resampler,
                                         hyper_param_object=LogisticClassifierHP(),
                                         params_grid=grid,
                                         model_persistence_manager=LocalCacheManager(cache_directory=cache_directory),  # noqa
                                         parallelization_cores=-1)
            return cart_base_model, rf_base_model, model_stacker, transformations, score_list, tuner

        cart_base_model, rf_base_model, model_stacker, transformations, score_list, tuner = create_tuner()
        # define the combinations of hyper-params that we want to evaluate
        tuner.tune(data_x=data.drop(columns=target_variable),
                   data_y=data[target_variable])

        # if these were actually reused each time they should still be None
        assert model_stacker._model_object is None
        assert all([x.value is None for x in score_list])
        assert all([x.state is None for x in transformations])
        assert cart_base_model.model._model_object is None
        assert rf_base_model.model._model_object is None

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_tuner_data.pkl'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=tuner.results.resampled_stats)  # noqa

        for fold in ['repeat{}_fold{}_'.format(0, x) for x in range(3)]:
            assert os.path.isfile(os.path.join(cache_directory, fold + 'train_meta.pkl'))
            assert os.path.isfile(os.path.join(cache_directory, fold + 'base_cart_criterion_gini_splitter_best_max_depth_None_min_samples_split_2_min_samples_leaf_1_min_weight_fraction_leaf_0.0_max_leaf_nodes_None_max_features_None.pkl'))  # noqa
            assert os.path.isfile(os.path.join(cache_directory, fold + 'base_random_forest_n_estimators_500_criterion_gini_max_features_None_max_depth_None_min_samples_split_2_min_samples_leaf_1_min_weight_fraction_leaf_0.0_max_leaf_nodes_None_min_impurity_decrease_0.0_bootstrap_True_oob_score_False.pkl'))  # noqa
            stacker_file = 'ModelStacker_LogisticClassifier_penaltyl2_regularization_inverse{}_solverliblinear.pkl'  # noqa
            for param in regularization_inverse:
                assert os.path.isfile(os.path.join(cache_directory, fold + stacker_file.format(float(param))))

        ######################################################################################################
        # repeat same thing but now files are cached
        ######################################################################################################
        cart_base_model, rf_base_model, model_stacker, transformations, score_list, tuner = create_tuner()
        # define the combinations of hyper-params that we want to evaluate
        tuner.tune(data_x=data.drop(columns=target_variable),
                   data_y=data[target_variable])

        # if these were actually reused each time they should still be None
        assert model_stacker._model_object is None
        assert all([x.value is None for x in score_list])
        assert all([x.state is None for x in transformations])
        assert cart_base_model.model._model_object is None
        assert rf_base_model.model._model_object is None

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_ModelWrappers/test_ModelStacker_tuner_data.pkl'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=tuner.results.resampled_stats)  # noqa

        for fold in ['repeat{}_fold{}_'.format(0, x) for x in range(3)]:
            assert os.path.isfile(os.path.join(cache_directory, fold + 'train_meta.pkl'))
            assert os.path.isfile(os.path.join(cache_directory, fold + 'base_cart_criterion_gini_splitter_best_max_depth_None_min_samples_split_2_min_samples_leaf_1_min_weight_fraction_leaf_0.0_max_leaf_nodes_None_max_features_None.pkl'))  # noqa
            assert os.path.isfile(os.path.join(cache_directory, fold + 'base_random_forest_n_estimators_500_criterion_gini_max_features_None_max_depth_None_min_samples_split_2_min_samples_leaf_1_min_weight_fraction_leaf_0.0_max_leaf_nodes_None_min_impurity_decrease_0.0_bootstrap_True_oob_score_False.pkl'))  # noqa
            stacker_file = 'ModelStacker_LogisticClassifier_penaltyl2_regularization_inverse{}_solverliblinear.pkl'  # noqa
            for param in regularization_inverse:
                assert os.path.isfile(os.path.join(cache_directory, fold + stacker_file.format(float(param))))

        shutil.rmtree(cache_directory)

    def test_ModelDefaults(self):
        ######################################################################################################
        # Regression
        ######################################################################################################
        default_models = ModelDefaults.get_regression_models(4)
        descriptions = [x.description for x in default_models]
        # ensure unique descriptions
        assert len(set(descriptions)) == len(descriptions)  # i.e. # of unique values equals number of values

        ######################################################################################################
        # Two-class classifiction
        ######################################################################################################
        default_models = ModelDefaults.get_twoclass_classification_models(4)
        descriptions = [x.description for x in default_models]
        # ensure unique descriptions
        assert len(set(descriptions)) == len(descriptions)  # i.e. # of unique values equals number of values

        ######################################################################################################
        # multi-class classifiction
        ######################################################################################################
        default_models = ModelDefaults.get_multiclass_classification_models(4)
        descriptions = [x.description for x in default_models]
        # ensure unique descriptions
        assert len(set(descriptions)) == len(descriptions)  # i.e. # of unique values equals number of values
