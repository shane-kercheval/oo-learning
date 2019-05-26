import math
import os
import pickle
from math import isclose
from typing import Callable, Union
from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
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
        iris = pd.DataFrame(data=np.c_[iris['data'],
                                       pd.Categorical.from_codes(iris['target'],
                                                                 ['setosa', 'versicolor', 'virginica'],
                                                                 ordered=False)],
                            columns=[x[0:12].strip().replace(' ', '_') for x in iris['feature_names']] +
                                    ['species'])
        iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] = \
            iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].apply(pd.to_numeric)

        return iris

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
    def ensure_all_values_equal(data_frame1: pd.DataFrame,
                                data_frame2: pd.DataFrame,
                                check_column_types: bool = True):
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        # check that the types of the columns are all the same
        if check_column_types:
            assert all([x == y for x, y in zip(data_frame1.dtypes.values, data_frame2.dtypes.values)])
        assert all(data_frame1.columns.values == data_frame2.columns.values)
        assert all(data_frame1.index.values == data_frame2.index.values)
        numeric_col, cat_cols = OOLearningHelpers.get_columns_by_type(data_dtypes=data_frame1.dtypes)

        for col in numeric_col:
            # check if the values are close, or if they are both NaN
            assert all([isclose(x, y) or (math.isnan(x) and math.isnan(y))
                        for x, y in zip(data_frame1[col].values, data_frame2[col].values)])

        for col in cat_cols:
            # if the two strings aren't equal, but also aren't 'nan', it will cause a problem because
            # isnan will try to convert the string to a number, but it will fail with TypeError, so have to
            # ensure both values are a number before we check that they are nan.
            assert all([x == y or (is_number(x) and is_number(y) and math.isnan(x) and math.isnan(y))
                        for x, y in zip(data_frame1[col].values, data_frame2[col].values)])

        return True

    @staticmethod
    def ensure_all_values_equal_from_file(file, expected_dataframe, check_column_types: bool = True):
        # with open(file, 'wb') as output:
        #     pickle.dump(expected_dataframe, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            found_dataframe = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=found_dataframe,
                                                      data_frame2=expected_dataframe,
                                                      check_column_types=check_column_types)

    # noinspection PyTypeChecker
    @staticmethod
    def ensure_series_equal_from_file(file, expected_series):
        # with open(file, 'wb') as output:
        #     pickle.dump(expected_series, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            found_series = pickle.load(saved_object)
            assert all(found_series.index.values == expected_series.index.values)
            assert all([isclose(x, y) for x, y in zip(found_series.values, expected_series.values)])

    @staticmethod
    def ensure_values_numeric_dictionary(dictionary_1, dictionary_2):
        assert set(dictionary_1.keys()) == set(dictionary_2.keys())
        assert all([isclose(dictionary_1[x], dictionary_2[x]) for x in dictionary_1.keys()])  # noqa

    @staticmethod
    def check_plot(file_name: str, get_plot_function: Callable, set_size_w_h: Union[tuple, None] = (11, 7)):
        def clear():
            plt.gcf().clear()
            plt.cla()
            plt.clf()
            plt.close()

        clear()
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory(file_name))
        if os.path.isfile(file):
            os.remove(file)
        assert os.path.isfile(file) is False
        get_plot_function()
        if set_size_w_h is not None:
            fig = plt.gcf()
            fig.set_size_inches(set_size_w_h[0], set_size_w_h[1])
        plt.savefig(file)
        clear()
        assert os.path.isfile(file)

    @staticmethod
    def assert_hyper_params_match(hyper_params: HyperParamsBase,
                                  model: ModelWrapperBase,
                                  mapping: dict = None,
                                  remove_keys: list = None):
        """
        :param hyper_params:
        :param model:
        :param mapping: when the hyper-parameter names don't between the HyperParamsBase object and the model,
            provide a dictionary with the HyperParamsBase name as the key, and the model's name as the value
        :param remove_keys: the keys to remove before checking
        :return:
        """
        subset = hyper_params.params_dict
        superset = model.model_object.get_params()

        TestHelper.assert_hyper_params_match_2(subset=subset,
                                               superset=superset,
                                               mapping=mapping,
                                               remove_keys=remove_keys)

    @staticmethod
    def assert_hyper_params_match_2(subset: dict,
                                    superset: dict,
                                    mapping: dict = None,
                                    remove_keys: list = None):
        """
        :param subset:
        :param superset:
        :param mapping: when the hyper-parameter names don't between the HyperParamsBase object and the model,
            provide a dictionary with the HyperParamsBase name as the key, and the model's name as the value
        :param remove_keys: the keys to remove before checking
        :return:
        """
        subset = subset.copy()
        superset = superset.copy()

        if remove_keys is not None:
            for key in remove_keys:
                del subset[key]

        if mapping is not None:
            subset = dict((mapping[key], value) if key in mapping else (key, value)
                          for (key, value) in subset.items())

        assert all(item in superset.items() for item in subset.items())

    @staticmethod
    def save_string(value: str, file: str):
        with open(TestHelper.ensure_test_directory(file), "w") as text_file:
            print(value, file=text_file)

    @staticmethod
    def save_df(df: pd.DataFrame, file: str):
        with open(TestHelper.ensure_test_directory(file), "w") as text_file:
            print(df.to_string(), file=text_file)
