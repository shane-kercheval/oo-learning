from math import isclose

import numpy as np
import pandas as pd

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


class MockFailedTransformer(TransformerBase):
    """
    This should not work because, for example, _fit_definition should set self._state
    """
    def _fit_definition(self, data_x) -> dict:
        pass

    def _transform_definition(self, data_x, state):
        return None


class MockSuccessTransformer(TransformerBase):
    """
    This should not work because, for example, _fit_definition should set self._state
    """
    def _fit_definition(self, data_x) -> dict:
        return {'junk_key': 'junk_value'}

    def _transform_definition(self, data_x, state):
        return data_x


# noinspection SpellCheckingInspection,PyMethodMayBeStatic, PyTypeChecker
class TransformerTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_base_class(self):
        ######################################################################################################
        # test the functionality of the logic in the base class, included for exampling, calling transform()
        # before fit(), calling fit() without stating the _state field, calling fit twice, etc.
        ######################################################################################################
        data = TestHelper.get_insurance_data()
        failing_transformer = MockFailedTransformer()

        self.assertRaises(AssertionError,
                          lambda: failing_transformer.transform(data_x=data))  # have not fitted data
        self.assertRaises(AssertionError,
                          lambda: failing_transformer.fit(data_x=data))  # fit does not set _state

        succeeding_transformer = MockSuccessTransformer()
        assert succeeding_transformer.state is None
        # have not fitted data
        self.assertRaises(AssertionError, lambda: succeeding_transformer.transform(data_x=data))
        succeeding_transformer.fit(data_x=data)  # test class sets _state
        assert len(succeeding_transformer.state) == 1
        assert succeeding_transformer.state['junk_key'] == 'junk_value'
        new_data = succeeding_transformer.transform(data_x=data)  # should work now
        assert new_data is not None
        # can't call fit() twice
        self.assertRaises(AssertionError, lambda: succeeding_transformer.fit(data_x=data))

    # noinspection PyTypeChecker
    def test_transformations_ImputationTransformer(self):
        ######################################################################################################
        # Test ImputationTransformer by using housing data, removing values, and then checking that the values
        # computed and filled correctly
        # Ensure future (e.g. test) datasets are also filled correctly
        ######################################################################################################
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_test_regression(data, target_variable)

        imputation_transformer = ImputationTransformer(categoric_imputation_function=None)
        # ensure that we are forced to call `fit` first
        self.assertRaises(AssertionError, lambda: imputation_transformer.transform(data_x=data))
        training_set_copy = training_set.copy()  # copy so that we don't change original training_set
        # remove a few values from a categorical column to make sure imputation doesn't include it

        training_set_number_nulls = training_set.isnull().values.sum()  # get the original # of None's
        expected_number_nulls = 171
        assert training_set_number_nulls == expected_number_nulls
        # introduce None's into a categorical variable
        training_set_copy.loc[0, 'ocean_proximity'] = None
        training_set_copy.loc[5, 'ocean_proximity'] = None
        training_set_copy.loc[60, 'ocean_proximity'] = None
        training_set_copy.loc[8788, 'ocean_proximity'] = None

        # make sure we get original + 4 number of Nones
        assert training_set_copy.isnull().values.sum() == training_set_number_nulls + 4
        # run fit_trasnform
        transformed_training_data = imputation_transformer.fit_transform(data_x=training_set_copy)

        # ensure that if we try to call fit_transform again we get an assertion error
        self.assertRaises(AssertionError,
                          lambda: imputation_transformer.fit_transform(data_x=training_set_copy))
        # ensure training_set_copy wasn't affected
        assert training_set_copy.isnull().values.sum() == expected_number_nulls + 4
        assert transformed_training_data.isnull().values.sum() == 4  # ensure all but 4 None's were replaced
        # ensure all the correct values/columns were imputed
        assert imputation_transformer.state == {'households': 410.0,
                                                'housing_median_age': 29.0,
                                                'latitude': 34.26,
                                                'longitude': -118.495,
                                                'median_income': 3.5318,
                                                'population': 1165,
                                                'total_bedrooms': 435.0,
                                                'total_rooms': 2125.0}
        # make sure we calculated the correct imputed value (median)
        imputed_median_total_bedrooms = imputation_transformer.state['total_bedrooms']
        assert training_set_copy['total_bedrooms'].median() == imputed_median_total_bedrooms

        # get the original indexes of null, and make sure the new dataset has the imputed value at all indexes
        indexes_of_null = np.where(training_set_copy['total_bedrooms'].isnull().values)[0]
        replaced_values = transformed_training_data.iloc[indexes_of_null]['total_bedrooms']
        assert all(replaced_values == imputed_median_total_bedrooms)

        # get indexes that were not null and make sure they haven't changed.
        indexes_not_null = [x for x in np.arange(0, len(training_set_copy), 1) if x not in indexes_of_null]
        assert all(training_set_copy.iloc[indexes_not_null]['total_bedrooms'] ==
                   transformed_training_data.iloc[indexes_not_null]['total_bedrooms'])

        # same for test set
        expected_number_nulls = 36
        assert test_set.isnull().values.sum() == expected_number_nulls
        transformed_test_data = imputation_transformer.transform(data_x=test_set)
        assert test_set.isnull().values.sum() == expected_number_nulls
        # make sure there are no
        assert transformed_test_data.isnull().values.sum() == 0
        # there is no reason the state should have changed
        assert imputation_transformer.state == {'households': 410.0,
                                                'housing_median_age': 29.0,
                                                'latitude': 34.26,
                                                'longitude': -118.495,
                                                'median_income': 3.5318,
                                                'population': 1165,
                                                'total_bedrooms': 435.0,
                                                'total_rooms': 2125.0}

        # get the original indexes of null, and make sure the new dataset has the imputed value at all indexes
        indexes_of_null = np.where(test_set['total_bedrooms'].isnull().values)[0]
        replaced_values = transformed_test_data.iloc[indexes_of_null]['total_bedrooms']
        assert all(replaced_values == imputed_median_total_bedrooms)

        # get indexes that were not null and make sure they haven't changed.
        indexes_not_null = [x for x in np.arange(0, len(test_set), 1) if x not in indexes_of_null]
        assert all(test_set.iloc[indexes_not_null]['total_bedrooms'] ==
                   transformed_test_data.iloc[indexes_not_null]['total_bedrooms'])

    def test_transformations_DummyEncodeTransformer(self):
        data = TestHelper.get_insurance_data()

        new_dummy_columns = ['age', 'bmi', 'children', 'expenses', 'sex_male', 'smoker_yes',
                             'region_northwest', 'region_southeast', 'region_southwest']
        new_one_hot_columns = ['age', 'bmi', 'children', 'expenses', 'sex_female', 'sex_male', 'smoker_no',
                               'smoker_yes', 'region_northeast', 'region_northwest',
                               'region_southeast', 'region_southwest']
        expected_state = {'region': ['northeast', 'northwest', 'southeast', 'southwest'],
                          'sex': ['female', 'male'],
                          'smoker': ['no', 'yes']}
        # Test DUMMY
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY)
        dummy_transformer.fit(data_x=data)

        assert dummy_transformer._columns_to_reindex == new_dummy_columns
        assert dummy_transformer.state == expected_state

        new_dummy_data = dummy_transformer.transform(data_x=data)
        assert new_dummy_data is not None
        assert len(new_dummy_data) == len(data)
        assert all(new_dummy_data.columns.values == new_dummy_columns)

        # Test ONE HOT
        one_hot_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.ONE_HOT)
        one_hot_transformer.fit(data_x=data)
        assert one_hot_transformer._columns_to_reindex == new_one_hot_columns
        assert one_hot_transformer.state == expected_state

        new_encoded_data = one_hot_transformer.transform(data_x=data)
        assert new_encoded_data is not None
        assert len(new_encoded_data) == len(data)
        assert all(new_encoded_data.columns.values == new_one_hot_columns)

        # test the values of one hot encoding,
        def assert_all_encoded_values(column_name):
            original_value = data[column_name][index]

            for value in expected_state[column_name]:  # check that the new values equal 0 or 1
                column = column_name + '_' + value
                expected_value = 1 if value == original_value else 0  # if old value == original value, then 1
                assert new_encoded_data[column][index] == expected_value

        for index in range(0, len(data)):  # cycle through each row, check the values of each dummy variable
            assert_all_encoded_values('region')
            assert_all_encoded_values('sex')
            assert_all_encoded_values('smoker')

        # now that we have tested all columns (one hot), ensure that the one hot columns match the
        # dummy columns that weren't dropped
        assert all(new_dummy_data['region_northwest'] == new_encoded_data['region_northwest'])
        assert all(new_dummy_data['region_southeast'] == new_encoded_data['region_southeast'])
        assert all(new_dummy_data['region_southwest'] == new_encoded_data['region_southwest'])
        assert all(new_dummy_data['sex_male'] == new_encoded_data['sex_male'])
        assert all(new_dummy_data['smoker_yes'] == new_encoded_data['smoker_yes'])

        # now test values when transforming individual rows (ONE HOT)
        for index in range(0, 50):  # test first 50 rows
            transformed_data = one_hot_transformer.transform(data_x=data.iloc[[index]])
            assert all(transformed_data.columns.values == new_encoded_data.iloc[[index]].columns.values)
            assert all(transformed_data == new_encoded_data.iloc[[index]])

            # now test values when transforming individual rows (ONE HOT)
        for index in range(0, 50):  # test first 50 rows
            transformed_data = dummy_transformer.transform(data_x=data.iloc[[index]])
            assert all(transformed_data.columns.values == new_dummy_data.iloc[[index]].columns.values)
            assert all(transformed_data == new_dummy_data.iloc[[index]])

        # TEST DATA WITH *NO* CATEGORICAL VALUES
        data = TestHelper.get_cement_data()
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY)
        dummy_transformer.fit(data_x=data)
        new_data = dummy_transformer.transform(data_x=data)
        assert all(new_data.columns.values == data.columns.values)

    def test_CategoricConverterTransformer(self):
        data = TestHelper.get_titanic_data()
        test_splitter = ClassificationStratifiedDataSplitter(test_ratio=0.05)
        training_indexes, test_indexes = test_splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        # the test data contains only a subset of the unique values for SibSp and Parch, so we can test that
        # the transformed data contains all categories in the Categorical object
        test_data = data.iloc[test_indexes]

        columns_to_convert = ['Pclass', 'SibSp', 'Parch']
        transformer = CategoricConverterTransformer(columns=columns_to_convert)
        transformer.fit(train_data)
        assert test_data.Pclass.dtype.name != 'category'
        assert test_data.SibSp.dtype.name != 'category'
        assert test_data.Parch.dtype.name != 'category'
        new_data = transformer.transform(test_data)
        assert new_data.Pclass.dtype.name == 'category'
        assert new_data.SibSp.dtype.name == 'category'
        assert new_data.Parch.dtype.name == 'category'

        assert all([new_data[x].dtype.name != 'category'
                    for x in new_data.columns.values
                    if x not in columns_to_convert])

        # check state has all correct values
        assert all([transformer.state[x] == sorted(data[x].dropna().unique().tolist())
                    for x in columns_to_convert])

        # check Categorical objects have correct values
        assert all([new_data[x].cat.categories.values.tolist() == sorted(data[x].dropna().unique().tolist())
                    for x in columns_to_convert])

    def test_transformations_TitanicDataset_test_various_conditions(self):
        """
        Test:
        -   At times there will be numeric columns that are actually categorical (e.g. 0/1)
                that we will need to specify
        -   Categoric imputation
        -   Test imputing only categoric, or only numeric
        -   test full pipeline
        """
        data = TestHelper.get_titanic_data()
        test_splitter = ClassificationStratifiedDataSplitter(test_ratio=0.05)
        training_indexes, test_indexes = test_splitter.split(target_values=data.Survived)
        train_data = data.iloc[training_indexes]
        # the test data contains only a subset of the unique values for SibSp and Parch, so we can test that
        # the transformed data contains all categories in e.g. the dummy columns
        test_data = data.iloc[test_indexes]

        indexes_of_null_train = {column: train_data[column].isnull().values for column in data.columns.values}
        indexes_of_null_test = {column: test_data[column].isnull().values for column in data.columns.values}

        pipeline = TransformerPipeline(transformations=[RemoveColumnsTransformer(['PassengerId',
                                                                                  'Name',
                                                                                  'Ticket',
                                                                                  'Cabin']),
                                                        CategoricConverterTransformer(['Pclass',
                                                                                       'SibSp',
                                                                                       'Parch']),
                                                        ImputationTransformer(),
                                                        DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)])

        transformed_training = pipeline.fit_transform(data_x=train_data)
        # transformed_training.to_csv('~/Desktop/trans_titanic.csv')
        assert transformed_training.columns.values.tolist() == \
            ['Survived', 'Age', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
             'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0',
             'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Embarked_C', 'Embarked_Q',
             'Embarked_S']
        assert train_data.isnull().sum().sum() > 0
        assert transformed_training.isnull().sum().sum() == 0

        expected_imputed_age = train_data.Age.median()

        assert all(transformed_training[indexes_of_null_train['Age']]['Age'] == expected_imputed_age)
        # imputed values for categorical will be found in the dummy variables
        # Mode for embarked is `S`, so all rows that had nulls should have Embarked_S column =1 and others = 0
        assert all(transformed_training[indexes_of_null_train['Embarked']]['Embarked_S'] == 1)
        assert all(transformed_training[indexes_of_null_train['Embarked']]['Embarked_C'] == 0)
        assert all(transformed_training[indexes_of_null_train['Embarked']]['Embarked_Q'] == 0)

        transformed_test = pipeline.transform(data_x=test_data)
        assert transformed_test.columns.values.tolist() == transformed_training.columns.values.tolist()
        assert test_data.isnull().sum().sum() > 0
        assert transformed_test.isnull().sum().sum() == 0

        assert all(transformed_test[indexes_of_null_test['Age']]['Age'] == expected_imputed_age)
        # no missing values for embarked to test

    def test_RemoveColumnsTransformer(self):
        data = TestHelper.get_insurance_data()

        def test_remove_columns(columns_to_remove):
            remove_column_transformer = RemoveColumnsTransformer(columns=columns_to_remove)
            new_data = remove_column_transformer.fit_transform(data_x=data)

            assert all(new_data.columns.values ==
                       [column for column in data.columns.values if column not in columns_to_remove])

        self.assertRaises(ValueError, lambda: test_remove_columns(columns_to_remove=['expensess']))

        # test removing each individual columns
        for x in data.columns.values:
            test_remove_columns(columns_to_remove=[x])

        # test removing incrementally more columns
        for index in range(1, len(data.columns.values)):
            test_remove_columns(columns_to_remove=list(data.columns.values[0:(index+1)]))

    def test_StatelessTransformer(self):
        """
        Create a StatelessTransformer that does the same thing as RemoveColumnsTransformer so that we can
        duplicate the test and ensure we get the same values. i.e. we should be indifferent towards methods
        """
        data = TestHelper.get_insurance_data()

        def test_remove_columns(columns_to_remove):
            def remove_columns_helper(data_to_transform: pd.DataFrame):
                return data_to_transform.drop(columns_to_remove, axis=1)

            remove_column_transformer = StatelessTransformer(custom_function=remove_columns_helper)
            new_data = remove_column_transformer.fit_transform(data_x=data)

            assert all(new_data.columns.values ==
                       [column for column in data.columns.values if column not in columns_to_remove])

        self.assertRaises(ValueError, lambda: test_remove_columns(columns_to_remove=['expensess']))

        # test removing each individual columns
        for x in data.columns.values:
            test_remove_columns(columns_to_remove=[x])

        # test removing incrementally more columns
        for index in range(1, len(data.columns.values)):
            test_remove_columns(columns_to_remove=list(data.columns.values[0:(index+1)]))

    def test_CenterScaleTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_test_regression(data, target_variable)

        transformer = CenterScaleTransformer()

        # ensure that we are forced to call `fit` first
        self.assertRaises(AssertionError, lambda: transformer.transform(data_x=data))

        transformed_training = transformer.fit_transform(data_x=training_set.copy())

        assert isclose(transformer._state['averages']['longitude'], training_set['longitude'].mean())
        assert isclose(transformer._state['averages']['latitude'], training_set['latitude'].mean())
        assert isclose(transformer._state['averages']['housing_median_age'], training_set['housing_median_age'].mean())  # noqa
        assert isclose(transformer._state['averages']['total_rooms'], training_set['total_rooms'].mean())
        assert isclose(transformer._state['averages']['total_bedrooms'], training_set['total_bedrooms'].mean())  # noqa
        assert isclose(transformer._state['averages']['population'], training_set['population'].mean())
        assert isclose(transformer._state['averages']['households'], training_set['households'].mean())
        assert isclose(transformer._state['averages']['median_income'], training_set['median_income'].mean())

        assert isclose(transformer._state['standard_deviations']['longitude'], training_set['longitude'].std())  # noqa
        assert isclose(transformer._state['standard_deviations']['latitude'], training_set['latitude'].std())
        assert isclose(transformer._state['standard_deviations']['housing_median_age'], training_set['housing_median_age'].std())  # noqa
        assert isclose(transformer._state['standard_deviations']['total_rooms'], training_set['total_rooms'].std())  # noqa
        assert isclose(transformer._state['standard_deviations']['total_bedrooms'], training_set['total_bedrooms'].std())  # noqa
        assert isclose(transformer._state['standard_deviations']['population'], training_set['population'].std())  # noqa
        assert isclose(transformer._state['standard_deviations']['households'], training_set['households'].std())  # noqa
        assert isclose(transformer._state['standard_deviations']['median_income'], training_set['median_income'].std())  # noqa

        # all transformed columns of the training set should have a mean of zero and a standard deviation of 1
        assert isclose(round(transformed_training['longitude'].mean(), 14), 0)
        assert isclose(round(transformed_training['latitude'].mean(), 14), 0)
        assert isclose(round(transformed_training['housing_median_age'].mean(), 14), 0)
        assert isclose(round(transformed_training['total_rooms'].mean(), 14), 0)
        assert isclose(round(transformed_training['total_bedrooms'].mean(), 14), 0)
        assert isclose(round(transformed_training['population'].mean(), 14), 0)
        assert isclose(round(transformed_training['households'].mean(), 14), 0)
        assert isclose(transformed_training['longitude'].std(), 1)
        assert isclose(transformed_training['latitude'].std(), 1)
        assert isclose(transformed_training['housing_median_age'].std(), 1)
        assert isclose(transformed_training['total_rooms'].std(), 1)
        assert isclose(transformed_training['total_bedrooms'].std(), 1)
        assert isclose(transformed_training['households'].std(), 1)
        assert isclose(transformed_training['population'].std(), 1)

        # mean and standard deviation of the test is not necessarily but should be close to 0/1
        # because we are using the values fitted from the training set
        transformed_test = transformer.transform(data_x=test_set.copy())

        assert isclose(round(transformed_test['longitude'].mean(), 1), 0)  # less precision, but ~0
        assert isclose(round(transformed_test['latitude'].mean(), 1), 0)  # less precision, but ~0
        assert isclose(round(transformed_test['housing_median_age'].mean(), 1), 0)  # less precision, but ~0
        assert isclose(round(transformed_test['total_rooms'].mean(), 1), 0)  # less precision, but ~0
        assert isclose(round(transformed_test['total_bedrooms'].mean(), 1), 0)  # less precision, but ~0
        assert isclose(round(transformed_test['population'].mean(), 1), 0)  # less precision, but ~0
        assert isclose(round(transformed_test['households'].mean(), 1), 0)  # less precision, but ~0
        assert isclose(round(transformed_test['longitude'].std(), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['latitude'].std(), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['housing_median_age'].std(), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['total_rooms'].std(), 0), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['total_bedrooms'].std(), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['households'].std(), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['population'].std(), 0), 1)  # less precision, but ~1

    # TODO: generating warning
    def test_BoxCoxTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_test_regression(data, target_variable)

        # training_set.housing_median_age.hist()
        # training_set.total_rooms.hist()
        # training_set.total_bedrooms.hist()
        # training_set.population.hist()
        # training_set.households.hist()
        # training_set.median_income.hist()

        # fail because we are passing in a feature that doesn't exist
        transformer = BoxCoxTransformer(features=['invalid_feature', 'total_rooms'])
        self.assertRaises(AssertionError, lambda: transformer.fit_transform(data_x=training_set))
        # fail because we are passing in a feature that has negative values
        transformer = BoxCoxTransformer(features=['longitude', 'housing_median_age', 'total_rooms'])
        self.assertRaises(NegativeValuesFoundError, lambda: transformer.fit_transform(data_x=training_set))
        # should work
        transformer = BoxCoxTransformer(features=['housing_median_age', 'total_rooms', 'total_bedrooms',
                                                  'population', 'households', 'median_income'])
        transformed_data = transformer.fit_transform(data_x=training_set)
        # ensure that non-specified columns didnt' change
        assert all(training_set.longitude == transformed_data.longitude)
        assert all(training_set.latitude == transformed_data.latitude)
        assert all(training_set.ocean_proximity == transformed_data.ocean_proximity)
        assert all(training_set.columns.values == transformed_data.columns.values)

        assert transformer.state == {'housing_median_age': 0.80574535816717296,
                                     'total_rooms': 0.22722010735989001,
                                     'total_bedrooms': 8.4721358117221772,
                                     'population': 0.24616608260632827,
                                     'households': 0.24939757927863923,
                                     'median_income': 0.083921974272447436}

        # transformed_data.housing_median_age.hist()
        # transformed_data.total_rooms.hist()
        # transformed_data.total_bedrooms.hist()
        # transformed_data.population.hist()
        # transformed_data.households.hist()
        # transformed_data.median_income.hist()

        transformed_data = transformer.transform(data_x=test_set)
        assert transformed_data is not None

        # transformed_data.housing_median_age.hist()
        # transformed_data.total_rooms.hist()
        # transformed_data.total_bedrooms.hist()
        # transformed_data.population.hist()
        # transformed_data.households.hist()
        # transformed_data.median_income.hist()

    # @unittest.skip("")
    # def test_RemoveNZPTransformer(self):
    #     pass

    def test_RemoveCorrelationsTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_test_regression(data, target_variable)

        # at a threshold of 0.95, only total_bedrooms should be removed
        expected_removed = ['total_bedrooms']
        transformer = RemoveCorrelationsTransformer(max_correlation_threshold=0.95)
        transformed_data = transformer.fit_transform(data_x=training_set)
        assert transformer.state['columns_to_remove'] == expected_removed
        assert list(transformed_data.columns.values) == [x for x in training_set.columns.values
                                                         if x not in expected_removed]
        transformed_data = transformer.transform(data_x=test_set)
        assert list(transformed_data.columns.values) == [x for x in training_set.columns.values
                                                         if x not in expected_removed]

        # at a threshold of 0.90, total_beedrooms, longtitude, total_rooms, and population should be removed
        expected_removed = ['total_bedrooms', 'longitude', 'total_rooms', 'population']
        transformer = RemoveCorrelationsTransformer(max_correlation_threshold=0.90)
        transformed_data = transformer.fit_transform(data_x=training_set)
        assert transformer.state['columns_to_remove'] == expected_removed
        assert list(transformed_data.columns.values) == [x for x in training_set.columns.values
                                                         if x not in expected_removed]
        transformed_data = transformer.transform(data_x=test_set)
        assert list(transformed_data.columns.values) == [x for x in training_set.columns.values
                                                         if x not in expected_removed]

    def test_transformations_TransformerPipeline(self):
        data = TestHelper.get_insurance_data()
        original_columns = data.columns.values
        expected_dummy_columns = ['age', 'bmi', 'children', 'expenses', 'sex_male', 'smoker_yes',
                                  'region_northwest', 'region_southeast', 'region_southwest']

        # test the case where no transformations are requested.
        transformer_pipeline = TransformerPipeline(transformations=None)
        # test merging with empty pipeline
        transformer_pipeline.append_pipeline(TransformerPipeline(transformations=None))
        assert transformer_pipeline.transformations is None
        non_transformed_data = transformer_pipeline.fit_transform(data_x=data)  # no transformations to fit
        assert non_transformed_data is not None
        assert all(non_transformed_data == data)

        # test creation with transformations as non-list
        self.assertRaises(AssertionError,
                          lambda: TransformerPipeline(transformations=DummyEncodeTransformer(
                            encoding=CategoricalEncoding.DUMMY)))

        # test with a single transformation
        dummy_pipeline = TransformerPipeline(
            transformations=[DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY)])
        self.assertRaises(NotImplementedError, dummy_pipeline.fit)
        assert dummy_pipeline.transformations is not None

        new_data = dummy_pipeline.fit_transform(data_x=data)
        assert all(new_data.columns.values == expected_dummy_columns)
        new_data2 = dummy_pipeline.transform(data_x=data)
        assert all(new_data2.columns.values == expected_dummy_columns)

        # cannot call fit_transform twice
        self.assertRaises(AssertionError, lambda: dummy_pipeline.fit_transform(data_x=data))

        # test with a multiple transformations
        expected_dummy_columns = ['bmi', 'children', 'expenses', 'sex_male', 'smoker_yes']
        transformations = [RemoveColumnsTransformer(columns=['age', 'region']),
                           DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY)]

        new_pipeline = TransformerPipeline(transformations=transformations)
        ######################################################################################################
        # First test `get_expected_columns()`, because it basically uses all the Transformations objects,
        # except that it first makes copies. So we need to make sure we can reuse all the Transformation
        # objects, since they can only be used once (which is enforced).
        ######################################################################################################
        columns = TransformerPipeline.get_expected_columns(transformations=transformations, data=data)
        assert columns == expected_dummy_columns
        ######################################################################################################
        # Now test the actual pipeline (fit_transform & transform)
        ######################################################################################################
        new_data = new_pipeline.fit_transform(data_x=data)
        assert len(data) == len(new_data)
        assert all(new_data.columns.values == expected_dummy_columns)
        assert all(data.columns.values == original_columns)

        new_data2 = new_pipeline.transform(data_x=data)
        assert len(data) == len(new_data2)
        assert all(new_data2.columns.values == expected_dummy_columns)
        assert all(data.columns.values == original_columns)
