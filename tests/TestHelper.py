import os
from math import isclose
from typing import Callable
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.splitters.RegressionStratifiedDataSplitter import RegressionStratifiedDataSplitter
from oolearning.splitters.ClassificationStratifiedDataSplitter import ClassificationStratifiedDataSplitter


class TestHelper:

    # noinspection SpellCheckingInspection
    @staticmethod
    def is_debugging():
        import inspect
        for frame in inspect.stack():
            if frame[1].endswith("pydevd.py"):
                return True

        return False

    @staticmethod
    def ensure_test_directory(path):
        return path if TestHelper.is_debugging() else 'tests/'+path

    @staticmethod
    def get_data(data_path):
        import os
        cwd_path = os.getcwd()
        if not TestHelper.is_debugging():
            data_path = 'tests/' + data_path
        os.path.join(cwd_path, data_path)

        return pd.read_csv(os.path.join(cwd_path, data_path))

    @staticmethod
    def get_housing_data():
        data = TestHelper.get_data(data_path='data/housing.csv')
        np.random.seed(42)
        data['temp_categorical'] = pd.cut(np.random.normal(size=len(data)), bins=[-20, -1, 0, 1, 20])

        return data

    @staticmethod
    def get_insurance_data():
        return TestHelper.get_data(data_path='data/insurance.csv')

    @staticmethod
    def get_credit_data():
        return TestHelper.get_data(data_path='data/credit.csv')

    @staticmethod
    def get_cement_data():
        return TestHelper.get_data(data_path='data/cement.csv')

    @staticmethod
    def get_titanic_data():
        return TestHelper.get_data(data_path='data/titanic.csv')

    @staticmethod
    def get_iris_data():
        iris = datasets.load_iris()
        # noinspection SpellCheckingInspection
        return pd.DataFrame(data=np.c_[iris['data'],
                                       pd.Categorical.from_codes(iris['target'],
                                                                 ['setosa', 'versicolor', 'virginica'],
                                                                 ordered=False)],
                            columns=[x[0:12].strip().replace(' ', '_') for x in iris['feature_names']] +
                                    ['species'])

    @staticmethod
    def get_insurance_expected_values():
        return TestHelper.get_data(data_path='data/insurance_expected_values.csv')

    @staticmethod
    def split_train_holdout_regression(data,
                                       target_variable,
                                       splitter=RegressionStratifiedDataSplitter(holdout_ratio=0.20)):
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])
        # return training data, training target data, holdout data, holdout target data
        return \
            data.iloc[training_indexes].drop(columns=target_variable),\
            data.iloc[training_indexes][target_variable].values,\
            data.iloc[holdout_indexes].drop(columns=target_variable),\
            data.iloc[holdout_indexes][target_variable].values

    @staticmethod
    def split_train_holdout_class(data,
                                  target_variable,
                                  splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.20)):
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])
        # return training data, training target data, holdout data, holdout target data
        return \
            data.iloc[training_indexes].drop(columns=target_variable),\
            data.iloc[training_indexes][target_variable].values,\
            data.iloc[holdout_indexes].drop(columns=target_variable),\
            data.iloc[holdout_indexes][target_variable].values

    # noinspection PyTypeChecker
    @staticmethod
    def ensure_all_values_equal(data_frame1, data_frame2):
        assert all(data_frame1.columns.values == data_frame2.columns.values)
        assert all(data_frame1.index.values == data_frame2.index.values)
        numeric_col, cat_cols = OOLearningHelpers.get_columns_by_type(data_dtypes=data_frame1.dtypes)

        for col in numeric_col:
            assert all([isclose(x, y) for x, y in zip(data_frame1[col].values, data_frame2[col].values)])

        for col in cat_cols:
            assert all([x == y for x, y in zip(data_frame1[col].values, data_frame2[col].values)])

        return True

    @staticmethod
    def check_plot(file_name: str, get_plot_function: Callable):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory(file_name))
        if os.path.isfile(file):
            os.remove(file)
        assert os.path.isfile(file) is False
        get_plot_function()
        fig = plt.gcf()
        fig.set_size_inches(11, 7)
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)
