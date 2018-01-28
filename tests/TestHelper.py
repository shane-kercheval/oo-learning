import numpy as np
import pandas as pd
from math import isclose
from oolearning.splitters.RegressionStratifiedDataSplitter import RegressionStratifiedDataSplitter
from oolearning.OOLearningHelpers import OOLearningHelpers


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
    def get_cement_data():
        return TestHelper.get_data(data_path='data/cement.csv')

    @staticmethod
    def get_titanic_data():
        return TestHelper.get_data(data_path='data/titanic.csv')

    @staticmethod
    def get_insurance_expected_values():
        return TestHelper.get_data(data_path='data/insurance_expected_values.csv')

    @staticmethod
    def split_train_test_regression(data,
                                    target_variable,
                                    test_splitter=RegressionStratifiedDataSplitter(test_ratio=0.20)):
        training_indexes, test_indexes = test_splitter.split(target_values=data[target_variable])

        # return training data, training target data, test data, test target data
        return \
            data.iloc[training_indexes].drop(target_variable, axis=1),\
            data.iloc[training_indexes][target_variable].values,\
            data.iloc[test_indexes].drop(target_variable, axis=1),\
            data.iloc[test_indexes][target_variable].values

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
