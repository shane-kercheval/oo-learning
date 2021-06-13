from math import isclose
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize

import numpy as np
import pandas as pd

from oolearning import *
from oolearning.model_wrappers.ModelExceptions import AlreadyExecutedError, NotExecutedError
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


class MockFailedTransformer(TransformerBase):
    """
    This should not work because, for example, _fit_definition should set self._state
    """

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x) -> dict:
        pass

    def _transform_definition(self, data_x, state):
        return None


class MockSuccessTransformer(TransformerBase):
    """
    This should not work because, for example, _fit_definition should set self._state
    """

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x) -> dict:
        return {'junk_key': 'junk_value'}

    def _transform_definition(self, data_x, state):
        return data_x


# noinspection SpellCheckingInspection,PyMethodMayBeStatic, PyTypeChecker
class TransformerTests(TimerTestCase):

    @staticmethod
    def check_non_nas(indexes_of_nas, column, dataset1, dataset2):
        # ensure non-na columns match
        # grab all the indexes, NOT in the indexes_of_na
        series_1 = dataset1.loc[~dataset1.index.isin(indexes_of_nas), column]
        series_2 = dataset2.loc[~dataset2.index.isin(indexes_of_nas), column]
        # ensure that the series` indexes (non-NAs) + indexes of NAs == the indexes of original dataset
        assert set(list(series_1.index.values) + list(indexes_of_nas)) == set(dataset1.index.values)
        assert set(list(series_2.index.values) + list(indexes_of_nas)) == set(dataset2.index.values)
        assert all(series_1.index.values == series_2.index.values)
        assert all(series_1 == series_2)

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

        self.assertRaises(NotExecutedError,
                          lambda: failing_transformer.transform(data_x=data))  # have not fitted data
        self.assertRaises(AssertionError,
                          lambda: failing_transformer.fit(data_x=data))  # fit does not set _state

        succeeding_transformer = MockSuccessTransformer()
        assert succeeding_transformer.state is None
        # have not fitted data
        self.assertRaises(NotExecutedError, lambda: succeeding_transformer.transform(data_x=data))
        succeeding_transformer.fit(data_x=data)  # test class sets _state
        assert len(succeeding_transformer.state) == 1
        assert succeeding_transformer.state['junk_key'] == 'junk_value'
        new_data = succeeding_transformer.transform(data_x=data)  # should work now
        assert new_data is not None
        # can't call fit() twice
        self.assertRaises(AlreadyExecutedError, lambda: succeeding_transformer.fit(data_x=data))

    # noinspection PyTypeChecker
    def test_transformations_ImputationTransformer(self):
        ######################################################################################################
        # Test ImputationTransformer by using housing data, removing values, and then checking that the values
        # computed and filled correctly
        # Ensure future (e.g. test) datasets are also filled correctly
        ######################################################################################################
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        imputation_transformer = ImputationTransformer(categoric_imputation_function=None)
        # ensure that we are forced to call `fit` first
        self.assertRaises(NotExecutedError, lambda: imputation_transformer.transform(data_x=data))
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
        assert all(transformed_training_data.index.values == training_set_copy.index.values)

        # ensure that if we try to call fit_transform again we get an assertion error
        self.assertRaises(AlreadyExecutedError,
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

    def test_transformations_ImputationTransformer_by_group(self):
        data = TestHelper.get_titanic_data()
        data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

        # prepare data with additional na values
        training_set, _, test_set, _ = TestHelper.split_train_holdout_class(data, 'Survived')

        assert training_set.isna().sum().sum() == 139

        indexes_of_na_age = training_set[training_set['Age'].isna()].index.values
        indexes_of_na_embarked = training_set[training_set['Embarked'].isna()].index.values

        # change 0's to NA for "Fare" column and confirm
        indexes_of_zero_fare = training_set[training_set.Fare == 0].index.values
        assert len(indexes_of_zero_fare) == 14
        training_set['Fare'].replace(0, np.nan, inplace=True)
        new_fare_values = training_set.loc[indexes_of_zero_fare, 'Fare'].isna().values
        assert len(new_fare_values) == 14
        assert all([x == True for x in new_fare_values])  # noqa

        # also, test out when Pclass is missing
        # set the value of Pclass to NaN for the first record where we also have a NaN Fare
        training_set.loc[indexes_of_zero_fare[0], 'Pclass'] = np.nan
        # let's also set Sex for this row to NA as well, to check categoric data
        training_set.loc[indexes_of_zero_fare[0], 'Sex'] = np.nan
        # give Parch a special value to check to make sure we didn't change this value
        training_set.loc[indexes_of_zero_fare[0], 'Parch'] = 1000
        assert np.isnan(training_set.loc[indexes_of_zero_fare[0], 'Pclass'])

        ######################################################################################################
        # column to group by as NUMERIC
        ######################################################################################################
        imputation_transformer = ImputationTransformer(group_by_column='Pclass')
        transformed_training_data = imputation_transformer.fit_transform(data_x=training_set)
        assert transformed_training_data.isna().sum().sum() == 0  # should be 0 NAs
        assert all(transformed_training_data.index.values == training_set.index.values)  # index should match
        assert all(transformed_training_data.columns.values == training_set.columns.values)  # columns match
        assert imputation_transformer.state == {'Pclass': 3.0,
                                                'Age': {1: 38.0, 2: 29.0, 3: 24.0, 'all': 28.5},
                                                'SibSp': {1: 0.0, 2: 0.0, 3: 0.0, 'all': 0.0},
                                                'Parch': {1: 0.0, 2: 0.0, 3: 0.0, 'all': 0.0},
                                                'Fare': {1: 61.679199999999994, 2: 15.0479, 3: 8.05,
                                                         'all': 14.5},
                                                'Sex': {1: 'male', 2: 'male', 3: 'male', 'all': 'male'},
                                                'Embarked': {1: 'S', 2: 'S', 3: 'S', 'all': 'S'}}
        state = imputation_transformer.state

        ######################################################################################################
        # Pclass - 1 NA, manually set
        ######################################################################################################
        # Pclass : indexes_of_zero_fare[0] was set to na and should have value of state['Pclass'],
        # since it was the "group by" column
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Pclass'] == state['Pclass']
        # Pclass : all NON-nas should have maching values (only NA is indexes_of_zero_fare[0])
        TransformerTests.check_non_nas(indexes_of_nas=[indexes_of_zero_fare[0]], column='Pclass',
                                       dataset1=transformed_training_data, dataset2=training_set)

        ######################################################################################################
        # Age
        ######################################################################################################
        # First, there is 1 case where Age is NA and Pclass is NA (at the first index where fare == 0), which
        # should result in using the median for all of the Age Data i.e. state['Age']['all']
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Age'] == state['Age']['all']
        # Next Lets check all of the indexes that were not NA, ensure they did not change
        TransformerTests.check_non_nas(indexes_of_nas=indexes_of_na_age, column='Age',
                                       dataset1=transformed_training_data, dataset2=training_set)

        # lets check the remaining indexes that were NA, except for the NA associated with Pclass == NA,
        # which we already checked
        indexes = [x for x in indexes_of_na_age if x != indexes_of_zero_fare[0]]
        former_nas = transformed_training_data.loc[indexes]
        assert all(former_nas[former_nas.Pclass == 3].Age == state['Age'][3])
        assert all(former_nas[former_nas.Pclass == 2].Age == state['Age'][2])
        assert all(former_nas[former_nas.Pclass == 1].Age == state['Age'][1])

        ######################################################################################################
        # SibSp - 0 NAs
        ######################################################################################################
        assert all(transformed_training_data.SibSp == training_set.SibSp)

        ######################################################################################################
        # Parch - 0 NAs
        ######################################################################################################
        assert all(transformed_training_data.Parch == training_set.Parch)

        ######################################################################################################
        # Fare - 14 NAs that were set from Fares that had value of 0
        ######################################################################################################
        # First, there is 1 case where Fare is NA and Pclass is NA, which should result in state['Sex']['all']
        # it is at the first index where fare == 0
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Fare'] == state['Fare']['all']
        # Next Lets check all of the indexes that were not NA, ensure they did not change
        TransformerTests.check_non_nas(indexes_of_nas=indexes_of_zero_fare, column='Fare',
                                       dataset1=transformed_training_data, dataset2=training_set)

        # lets check the remaining indexes that were NA, except for the NA associated with Pclass == NA,
        # which we already checked
        former_nas = transformed_training_data.loc[indexes_of_zero_fare[1:]]
        assert all(former_nas[former_nas.Pclass == 3].Fare == state['Fare'][3])
        assert all(former_nas[former_nas.Pclass == 2].Fare == state['Fare'][2])
        assert all(former_nas[former_nas.Pclass == 1].Fare == state['Fare'][1])

        ######################################################################################################
        # Sex - 1 NA manually set
        ######################################################################################################
        # there is 1 clase where Sex was manually set to NA, it is the index that Pclass is NA, which should
        # result in state['Sex']['all']
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Sex'] == state['Sex']['all']

        ######################################################################################################
        # Emabarked - 2 NAs
        ######################################################################################################
        former_nas = transformed_training_data.loc[indexes_of_na_embarked]
        # both Pclas == 1
        assert all(former_nas[former_nas.Pclass == 1].Embarked == state['Embarked'][1])

        ######################################################################################################
        # Test test_set
        ######################################################################################################
        # Age
        assert test_set.isna().sum().sum() == 40  # there are 40 NAs
        assert test_set.isna().sum().Age == 40  # and all NAs ae in Age
        transformed_test_data = imputation_transformer.transform(data_x=test_set)
        test_indexes_na_age = test_set[test_set['Age'].isna()].index.values
        assert len(test_indexes_na_age) == 40
        # Next Lets check all of the indexes that were not NA, ensure they did not change
        TransformerTests.check_non_nas(indexes_of_nas=test_indexes_na_age, column='Age',
                                       dataset1=transformed_test_data, dataset2=test_set)
        # lets check the remaining indexes that were NA, except for the NA associated with Pclass == NA,
        # which we already checked
        former_nas = transformed_test_data.loc[test_indexes_na_age]
        assert all(former_nas[former_nas.Pclass == 3].Age == state['Age'][3])
        assert all(former_nas[former_nas.Pclass == 2].Age == state['Age'][2])
        assert all(former_nas[former_nas.Pclass == 1].Age == state['Age'][1])

        ######################################################################################################
        # column to group by as STRING
        ######################################################################################################
        def transform_column_to_categorical(mapping, dataset, feature):
            actual_to_expected_mapping = dict(zip(mapping.keys(), np.arange(len(mapping))))
            codes = pd.Series(dataset[feature]).map(actual_to_expected_mapping).fillna(-1)
            return pd.Categorical.from_codes(codes.astype(int), mapping.values(), ordered=False)

        training_set['Pclass'] = transform_column_to_categorical(mapping={1: 'a', 2: 'b', 3: 'c'},
                                                                 dataset=training_set,
                                                                 feature='Pclass')

        test_set['Pclass'] = transform_column_to_categorical(mapping={1: 'a', 2: 'b', 3: 'c'},
                                                             dataset=test_set,
                                                             feature='Pclass')

        imputation_transformer = ImputationTransformer(group_by_column='Pclass')
        transformed_training_data = imputation_transformer.fit_transform(data_x=training_set)
        assert transformed_training_data.isna().sum().sum() == 0  # should be 0 NAs
        assert all(transformed_training_data.index.values == training_set.index.values)  # index should match
        assert all(transformed_training_data.columns.values == training_set.columns.values)  # columns match
        # state is the same except for Pclass is down with the categoric columns rather than with numeric
        assert imputation_transformer.state == {'Age': {'a': 38.0, 'b': 29.0, 'c': 24.0, 'all': 28.5},
                                                'SibSp': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'all': 0.0},
                                                'Parch': {'a': 0.0, 'b': 0.0, 'c': 0.0, 'all': 0.0},
                                                'Fare': {'a': 61.679199999999994, 'b': 15.0479, 'c': 8.05,
                                                         'all': 14.5},
                                                'Pclass': 'c',
                                                'Sex': {'a': 'male', 'b': 'male', 'c': 'male', 'all': 'male'},
                                                'Embarked': {'a': 'S', 'b': 'S', 'c': 'S', 'all': 'S'}}
        state = imputation_transformer.state

        ######################################################################################################
        # Pclass - 1 NA, manually set
        ######################################################################################################
        # Pclass : indexes_of_zero_fare[0] was set to na and should have value of state['Pclass'],
        # since it was the "group by" column
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Pclass'] == state['Pclass']
        # Pclass : all NON-nas should have maching values (only NA is indexes_of_zero_fare[0])
        TransformerTests.check_non_nas(indexes_of_nas=[indexes_of_zero_fare[0]], column='Pclass',
                                       dataset1=transformed_training_data, dataset2=training_set)

        ######################################################################################################
        # Age
        ######################################################################################################
        # First, there is 1 case where Age is NA and Pclass is NA, which should result in state['Sex']['all']
        # it is at the first index where fare == 0
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Age'] == state['Age']['all']
        # Next Lets check all of the indexes that were not NA, ensure they did not change
        TransformerTests.check_non_nas(indexes_of_nas=indexes_of_na_age, column='Age',
                                       dataset1=transformed_training_data, dataset2=training_set)

        # lets check the remaining indexes that were NA, except for the NA associated with Pclass == NA,
        # which we already checked
        indexes = [x for x in indexes_of_na_age if x != indexes_of_zero_fare[0]]
        former_nas = transformed_training_data.loc[indexes]
        assert all(former_nas[former_nas.Pclass == 'c'].Age == state['Age']['c'])
        assert all(former_nas[former_nas.Pclass == 'b'].Age == state['Age']['b'])
        assert all(former_nas[former_nas.Pclass == 'a'].Age == state['Age']['a'])

        ######################################################################################################
        # SibSp - 0 NAs
        ######################################################################################################
        assert all(transformed_training_data.SibSp == training_set.SibSp)

        ######################################################################################################
        # Parch - 0 NAs
        ######################################################################################################
        assert all(transformed_training_data.Parch == training_set.Parch)

        ######################################################################################################
        # Fare - 14 NAs that were set from Fares that had value of 0
        ######################################################################################################
        # First, there is 1 case where Fare is NA and Pclass is NA, which should result in state['Sex']['all']
        # it is at the first index where fare == 0
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Fare'] == state['Fare']['all']
        # Next Lets check all of the indexes that were not NA, ensure they did not change
        TransformerTests.check_non_nas(indexes_of_nas=indexes_of_zero_fare, column='Fare',
                                       dataset1=transformed_training_data, dataset2=training_set)

        # lets check the remaining indexes that were NA, except for the NA associated with Pclass == NA,
        # which we already checked
        former_nas = transformed_training_data.loc[indexes_of_zero_fare[1:]]
        assert all(former_nas[former_nas.Pclass == 'c'].Fare == state['Fare']['c'])
        assert all(former_nas[former_nas.Pclass == 'b'].Fare == state['Fare']['b'])
        assert all(former_nas[former_nas.Pclass == 'a'].Fare == state['Fare']['a'])

        ######################################################################################################
        # Sex - 1 NA manually set
        ######################################################################################################
        # there is 1 clase where Sex was manually set to NA, it is the index that Pclass is NA, which should
        # result in state['Sex']['all']
        assert transformed_training_data.loc[indexes_of_zero_fare[0], 'Sex'] == state['Sex']['all']

        ######################################################################################################
        # Emabarked - 2 NAs
        ######################################################################################################
        former_nas = transformed_training_data.loc[indexes_of_na_embarked]
        # both Pclas == 1
        assert all(former_nas[former_nas.Pclass == 'a'].Embarked == state['Embarked']['a'])

        ######################################################################################################
        # Test test_set
        ######################################################################################################
        # Age
        assert test_set.isna().sum().sum() == 40  # there are 40 NAs
        assert test_set.isna().sum().Age == 40  # and all NAs ae in Age
        transformed_test_data = imputation_transformer.transform(data_x=test_set)
        test_indexes_na_age = test_set[test_set['Age'].isna()].index.values
        assert len(test_indexes_na_age) == 40
        # Next Lets check all of the indexes that were not NA, ensure they did not change
        TransformerTests.check_non_nas(indexes_of_nas=test_indexes_na_age, column='Age',
                                       dataset1=transformed_test_data, dataset2=test_set)
        # lets check the remaining indexes that were NA, except for the NA associated with Pclass == NA,
        # which we already checked
        former_nas = transformed_test_data.loc[test_indexes_na_age]
        assert all(former_nas[former_nas.Pclass == 'c'].Age == state['Age']['c'])
        assert all(former_nas[former_nas.Pclass == 'b'].Age == state['Age']['b'])
        assert all(former_nas[former_nas.Pclass == 'a'].Age == state['Age']['a'])

    def test_transformations_ImputationTransformer_treat_zeros_as_na_include_columns(self):
        data = TestHelper.get_titanic_data()
        data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        # prepare data with additional na values
        training_set, _, test_set, _ = TestHelper.split_train_holdout_class(data, 'Survived')

        training_indexes_of_zero = training_set[training_set.Fare == 0].index.values
        test_indexes_of_zero = test_set[test_set.Fare == 0].index.values

        assert len(training_indexes_of_zero) == 14
        assert len(test_indexes_of_zero) == 1

        # include Embarked just to test categoric column (in columns_explicit)
        imputation_transformer = ImputationTransformer(group_by_column='Pclass',
                                                       treat_zeros_as_na=True,
                                                       columns_explicit=['Embarked', 'Fare'])

        transformed_training_data = imputation_transformer.fit_transform(data_x=training_set)
        assert all(transformed_training_data.index.values == training_set.index.values)  # index should match
        assert all(transformed_training_data.columns.values == training_set.columns.values)  # columns match
        assert imputation_transformer.state == {'Fare': {1: 61.679199999999994, 2: 15.0479, 3: 8.05,
                                                         'all': 14.5},
                                                'Embarked': {1: 'S', 2: 'S', 3: 'S', 'all': 'S'}}
        state = imputation_transformer.state

        # test that the changes (to 0) didn't affect the training set
        assert len(training_set[np.isnan(training_set.Fare)]) == 0

        # columns that didn't change
        assert all(transformed_training_data.Pclass == training_set.Pclass)
        assert all(transformed_training_data.Sex == training_set.Sex)
        assert all([x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(transformed_training_data.Age,
                                                                            training_set.Age)])
        assert all(transformed_training_data.SibSp == training_set.SibSp)
        assert all(transformed_training_data.Parch == training_set.Parch)

        # now check Fare
        TransformerTests.check_non_nas(indexes_of_nas=training_indexes_of_zero, column='Fare',
                                       dataset1=transformed_training_data, dataset2=training_set)

        assert not any(transformed_training_data.Fare == 0)
        former_nas = transformed_training_data.loc[training_indexes_of_zero]
        assert all(former_nas[former_nas.Pclass == 3].Fare == state['Fare'][3])
        assert all(former_nas[former_nas.Pclass == 2].Fare == state['Fare'][2])
        assert all(former_nas[former_nas.Pclass == 1].Fare == state['Fare'][1])

        # now check Embarked
        embarked_na_indexes = training_set[training_set.Embarked.isna()].index.values
        TransformerTests.check_non_nas(indexes_of_nas=embarked_na_indexes, column='Embarked',
                                       dataset1=transformed_training_data, dataset2=training_set)

        former_nas = transformed_training_data.loc[embarked_na_indexes]
        assert all(former_nas[former_nas.Pclass == 3].Embarked == state['Embarked'][3])
        assert all(former_nas[former_nas.Pclass == 2].Embarked == state['Embarked'][2])
        assert all(former_nas[former_nas.Pclass == 1].Embarked == state['Embarked'][1])

        ######################################################################################################
        # Test test_set
        ######################################################################################################
        transformed_test_data = imputation_transformer.transform(data_x=test_set)

        TransformerTests.check_non_nas(indexes_of_nas=test_indexes_of_zero, column='Fare',
                                       dataset1=transformed_test_data, dataset2=test_set)

        assert not any(transformed_test_data.Fare == 0)
        former_nas = transformed_test_data.loc[test_indexes_of_zero]
        assert all(former_nas[former_nas.Pclass == 3].Fare == state['Fare'][3])
        assert all(former_nas[former_nas.Pclass == 2].Fare == state['Fare'][2])
        assert all(former_nas[former_nas.Pclass == 1].Fare == state['Fare'][1])

    def test_transformations_ImputationTransformer_include_exclude_columns(self):
        data = TestHelper.get_titanic_data()
        # prepare data with additional na values
        training_set, _, test_set, _ = TestHelper.split_train_holdout_class(data, 'Survived')

        training_indexes_of_zero = training_set[training_set.Fare == 0].index.values
        test_indexes_of_zero = test_set[test_set.Fare == 0].index.values

        assert len(training_indexes_of_zero) == 14
        assert len(test_indexes_of_zero) == 1

        # include Embarked just to test categoric column (in columns_explicit)
        columns_to_ignore = [x for x in data.columns.values if x not in ['Embarked', 'Fare']]
        imputation_transformer = ImputationTransformer(group_by_column='Pclass',
                                                       treat_zeros_as_na=True,
                                                       columns_to_ignore=columns_to_ignore)

        transformed_training_data = imputation_transformer.fit_transform(data_x=training_set)
        assert all(transformed_training_data.index.values == training_set.index.values)  # index should match
        assert all(transformed_training_data.columns.values == training_set.columns.values)  # columns match
        assert imputation_transformer.state == {'Fare': {1: 61.679199999999994, 2: 15.0479, 3: 8.05,
                                                         'all': 14.5},
                                                'Embarked': {1: 'S', 2: 'S', 3: 'S', 'all': 'S'}}
        state = imputation_transformer.state

        # test that the changes (to 0) didn't affect the training set
        assert len(training_set[np.isnan(training_set.Fare)]) == 0

        # columns that didn't change
        assert all(transformed_training_data.Pclass == training_set.Pclass)
        assert all(transformed_training_data.Sex == training_set.Sex)
        # Age was ignored, so it should still have the same amount of NA values
        assert training_set['Age'].isna().sum() == transformed_training_data['Age'].isna().sum()
        assert all([x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(transformed_training_data.Age,
                                                                            training_set.Age)])
        assert all(transformed_training_data.SibSp == training_set.SibSp)
        assert all(transformed_training_data.Parch == training_set.Parch)

        # now check Fare
        TransformerTests.check_non_nas(indexes_of_nas=training_indexes_of_zero, column='Fare',
                                       dataset1=transformed_training_data, dataset2=training_set)

        assert not any(transformed_training_data.Fare == 0)
        former_nas = transformed_training_data.loc[training_indexes_of_zero]
        assert all(former_nas[former_nas.Pclass == 3].Fare == state['Fare'][3])
        assert all(former_nas[former_nas.Pclass == 2].Fare == state['Fare'][2])
        assert all(former_nas[former_nas.Pclass == 1].Fare == state['Fare'][1])

        # now check Embarked
        embarked_na_indexes = training_set[training_set.Embarked.isna()].index.values
        assert len(embarked_na_indexes) == 2
        TransformerTests.check_non_nas(indexes_of_nas=embarked_na_indexes, column='Embarked',
                                       dataset1=transformed_training_data, dataset2=training_set)

        former_nas = transformed_training_data.loc[embarked_na_indexes]
        assert all(former_nas[former_nas.Pclass == 3].Embarked == state['Embarked'][3])
        assert all(former_nas[former_nas.Pclass == 2].Embarked == state['Embarked'][2])
        assert all(former_nas[former_nas.Pclass == 1].Embarked == state['Embarked'][1])

        ######################################################################################################
        # Test test_set
        ######################################################################################################
        transformed_test_data = imputation_transformer.transform(data_x=test_set)

        TransformerTests.check_non_nas(indexes_of_nas=test_indexes_of_zero, column='Fare',
                                       dataset1=transformed_test_data, dataset2=test_set)

        assert not any(transformed_test_data.Fare == 0)
        former_nas = transformed_test_data.loc[test_indexes_of_zero]
        assert all(former_nas[former_nas.Pclass == 3].Fare == state['Fare'][3])
        assert all(former_nas[former_nas.Pclass == 2].Fare == state['Fare'][2])
        assert all(former_nas[former_nas.Pclass == 1].Fare == state['Fare'][1])

    def test_transformations_PolynomialFeaturesTransformer(self):
        data = TestHelper.get_insurance_data()
        target_variable = 'expenses'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)
        transformer = PolynomialFeaturesTransformer(degrees=2)
        transformer.fit(data_x=training_set)
        # "fitting" the data should not actually rely on the training data (i.e. it is not fitting the data,
        # capturing which columns to transform, so the transformed training data and test data should only
        # on the data in their own set
        assert transformer.state == {}  # nothing to retain
        new_train_data = transformer.transform(data_x=training_set)
        # check that indexes are retained
        assert all(new_train_data.index == training_set.index)
        assert all(new_train_data.columns.values == ['age', 'bmi', 'children', 'age^2', 'age bmi',
                                                     'age children', 'bmi^2', 'bmi children', 'children^2',
                                                     'sex', 'smoker', 'region'])
        # original columns
        assert all(new_train_data.age == training_set.age)
        assert all(new_train_data.bmi == training_set.bmi)
        assert all(new_train_data.children == training_set.children)
        assert all(new_train_data.sex == training_set.sex)
        assert all(new_train_data.smoker == training_set.smoker)
        assert all(new_train_data.region == training_set.region)
        # squared columns
        assert all(new_train_data['age^2'] == training_set.age**2)
        assert all(new_train_data['bmi^2'] == training_set.bmi ** 2)
        assert all(new_train_data['children^2'] == training_set.children ** 2)
        # interaction affects
        assert all(new_train_data['age bmi'] == training_set.age * training_set.bmi)
        assert all(new_train_data['age children'] == training_set.age * training_set.children)
        assert all(new_train_data['bmi children'] == training_set.bmi * training_set.children)

        # test transforming a new unseen dataset (shouldn't matter, again, results are not dependent on
        # training set)
        new_test_data = transformer.transform(data_x=test_set)
        # check that indexes are retained
        assert all(new_test_data.index == test_set.index)
        assert all(new_test_data.columns.values == ['age', 'bmi', 'children', 'age^2', 'age bmi',
                                                    'age children', 'bmi^2', 'bmi children', 'children^2',
                                                    'sex', 'smoker', 'region'])
        # original columns
        assert all(new_test_data.age == test_set.age)
        assert all(new_test_data.bmi == test_set.bmi)
        assert all(new_test_data.children == test_set.children)
        assert all(new_test_data.sex == test_set.sex)
        assert all(new_test_data.smoker == test_set.smoker)
        assert all(new_test_data.region == test_set.region)
        # squared columns
        assert all(new_test_data['age^2'] == test_set.age**2)
        assert all(new_test_data['bmi^2'] == test_set.bmi ** 2)
        assert all(new_test_data['children^2'] == test_set.children ** 2)
        # interaction affects
        assert all(new_test_data['age bmi'] == test_set.age * test_set.bmi)
        assert all(new_test_data['age children'] == test_set.age * test_set.children)
        assert all(new_test_data['bmi children'] == test_set.bmi * test_set.children)

    def test_transformations_DummyEncodeTransformer(self):
        data = TestHelper.get_insurance_data()
        # make sure encoding works when columns names have spaces
        data = data.rename(index=str, columns={'region': 'reg ion'})

        new_dummy_columns = ['age', 'bmi', 'children', 'expenses', 'sex_male', 'smoker_yes',
                             'reg ion_northwest', 'reg ion_southeast', 'reg ion_southwest']
        new_one_hot_columns = ['age', 'bmi', 'children', 'expenses', 'sex_female', 'sex_male', 'smoker_no',
                               'smoker_yes', 'reg ion_northeast', 'reg ion_northwest',
                               'reg ion_southeast', 'reg ion_southwest']
        expected_state = {'reg ion': ['northeast', 'northwest', 'southeast', 'southwest'],
                          'sex': ['female', 'male'],
                          'smoker': ['no', 'yes']}
        # Test DUMMY
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY)
        dummy_transformer.fit(data_x=data)

        assert dummy_transformer._columns_to_reindex == new_dummy_columns
        assert dummy_transformer.encoded_columns == ['sex_male', 'smoker_yes', 'reg ion_northwest',
                                                     'reg ion_southeast', 'reg ion_southwest']
        assert dummy_transformer.state == expected_state

        new_dummy_data = dummy_transformer.transform(data_x=data)
        assert all(new_dummy_data.index.values == data.index.values)
        assert new_dummy_data is not None
        assert len(new_dummy_data) == len(data)
        assert all(new_dummy_data.columns.values == new_dummy_columns)

        # Test ONE HOT
        one_hot_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.ONE_HOT)
        one_hot_transformer.fit(data_x=data)
        assert one_hot_transformer._columns_to_reindex == new_one_hot_columns
        assert one_hot_transformer.encoded_columns == ['sex_female', 'sex_male', 'smoker_no', 'smoker_yes',
                                                       'reg ion_northeast', 'reg ion_northwest',
                                                       'reg ion_southeast', 'reg ion_southwest']
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
            assert_all_encoded_values('reg ion')
            assert_all_encoded_values('sex')
            assert_all_encoded_values('smoker')

        # now that we have tested all columns (one hot), ensure that the one hot columns match the
        # dummy columns that weren't dropped
        assert all(new_dummy_data['reg ion_northwest'] == new_encoded_data['reg ion_northwest'])
        assert all(new_dummy_data['reg ion_southeast'] == new_encoded_data['reg ion_southeast'])
        assert all(new_dummy_data['reg ion_southwest'] == new_encoded_data['reg ion_southwest'])
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

    def test_transformations_DummyEncodeTransformer_choose_leaveout(self):
        data = TestHelper.get_insurance_data()
        # make sure encoding works when columns names have spaces
        data = data.rename(index=str, columns={'sex': 'se x'})

        new_dummy_columns = ['age', 'bmi', 'children', 'expenses', 'se x_female', 'smoker_yes',
                             'region_northeast', 'region_northwest', 'region_southwest']

        expected_state = {'region': ['northeast', 'northwest', 'southeast', 'southwest'],
                          'se x': ['female', 'male'],
                          'smoker': ['no', 'yes']}
        # Test DUMMY
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY,
                                                   leave_out_columns={'region': 'southeast',
                                                                      'se x': 'male'})
        dummy_transformer.fit(data_x=data)

        assert dummy_transformer._columns_to_reindex == new_dummy_columns
        assert dummy_transformer.encoded_columns == ['se x_female', 'smoker_yes', 'region_northeast',
                                                     'region_northwest', 'region_southwest']
        assert dummy_transformer.state == expected_state

        new_dummy_data = dummy_transformer.transform(data_x=data)
        assert all(new_dummy_data.index.values == data.index.values)
        assert new_dummy_data is not None
        assert len(new_dummy_data) == len(data)
        assert all(new_dummy_data.columns.values == new_dummy_columns)

        ######################################################################################################
        # test with different values
        ######################################################################################################
        data = TestHelper.get_insurance_data()
        # make sure encoding works when columns names have spaces
        data = data.rename(index=str, columns={'sex': 'se x'})

        new_dummy_columns = ['age', 'bmi', 'children', 'expenses', 'se x_male', 'smoker_yes',
                             'region_northeast', 'region_northwest', 'region_southeast']

        expected_state = {'region': ['northeast', 'northwest', 'southeast', 'southwest'],
                          'se x': ['female', 'male'],
                          'smoker': ['no', 'yes']}
        # Test DUMMY
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY,
                                                   leave_out_columns={'region': 'southwest',
                                                                      'smoker': 'no'}
                                                   )
        dummy_transformer.fit(data_x=data)

        assert dummy_transformer._columns_to_reindex == new_dummy_columns
        assert dummy_transformer.encoded_columns == ['se x_male', 'smoker_yes', 'region_northeast',
                                                     'region_northwest', 'region_southeast']
        assert dummy_transformer.state == expected_state

        new_dummy_data = dummy_transformer.transform(data_x=data)
        assert all(new_dummy_data.index.values == data.index.values)
        assert new_dummy_data is not None
        assert len(new_dummy_data) == len(data)
        assert all(new_dummy_data.columns.values == new_dummy_columns)

        # TEST DATA WITH *NO* CATEGORICAL VALUES
        data = TestHelper.get_cement_data()
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY)
        dummy_transformer.fit(data_x=data)
        new_data = dummy_transformer.transform(data_x=data)
        assert all(new_data.columns.values == data.columns.values)

        data = TestHelper.get_cement_data()
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.ONE_HOT)
        dummy_transformer.fit(data_x=data)
        new_data = dummy_transformer.transform(data_x=data)
        assert all(new_data.columns.values == data.columns.values)

    def test_transformations_DummyEncodeTransformer_peaking(self):
        # the problem is that when Encoding, specifically when resampling etc…. the data/Transformer is
        # fitted with a subset of values that it will eventually see, and if a rare value is not in the
        # dataset that is fitted, but shows up in a future dataset (i.e. during `transform`), then getting the
        # encoded columns would result in a dataset that contains columns that the model didn't see when
        # fitting, and therefore, doesn't know what to do with. So, before transforming, we will allow
        # the transformer to 'peak' at ALL the data.
        data = TestHelper.get_insurance_data()

        # columns without smoker_maybe
        wrong_columns = ['age', 'bmi', 'children', 'expenses', 'sex_female', 'sex_male', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']  # noqa
        expected_columns = ['age', 'bmi', 'children', 'expenses', 'sex_female', 'sex_male', 'smoker_maybe', 'smoker_no', 'smoker_yes', 'region_northeast', 'region_northwest', 'region_southeast', 'region_southwest']  # noqa
        expected_state = {'region': ['northeast', 'northwest', 'southeast', 'southwest'],
                          'sex': ['female', 'male'],
                          'smoker': ['maybe', 'no', 'yes']}

        # one person/observation has the value of `maybe`, all others are `yes`/`no`
        data.loc[0, 'smoker'] = 'maybe'

        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.ONE_HOT)
        dummy_transformer.fit(data_x=data.drop(index=0))  # the transformer doesn't know about the value
        # the data has the wrong columns because it doesn't know about the 'maybe' value
        assert dummy_transformer._columns_to_reindex == wrong_columns
        assert dummy_transformer.encoded_columns == ['sex_female', 'sex_male', 'smoker_no', 'smoker_yes',
                                                     'region_northeast', 'region_northwest',
                                                     'region_southeast', 'region_southwest']

        # now, when we try to fit with the data that has the unseen values, it will explode
        self.assertRaises(AssertionError, lambda: dummy_transformer.transform(data_x=data))

        # But if we peak at the data first, then fit only what, in a real scenario, would be the training data
        # then it should work. Note, let's use the TransformerPipeline, which would be overkill in a normal
        # scnario like this, but we will use it to get the benefit of testing it, and the peak function out.
        dummy_transformer = DummyEncodeTransformer(encoding=CategoricalEncoding.ONE_HOT)
        pipeline = TransformerPipeline(transformations=[dummy_transformer])
        # peak at all the data
        pipeline.peak(data_x=data)
        # fit on only the "train" dataset (and also transform)
        train_x_transformed = pipeline.fit_transform(data_x=data.drop(index=0))

        # ensure that, even though we call "fit" on the data with "maybe" it should still be in the expected
        # columns & state
        assert all(train_x_transformed.columns.values == expected_columns)
        assert dummy_transformer.state == expected_state

        # transform on only 1 row, the row that was not "fitted"; hint, it should not fucking explode
        # NOTE: have to slice (i.e. [0:1] since iloc of 1 row returns a Series, not a DataFrame
        test_x_transformed = pipeline.transform(data_x=data.iloc[0:1])
        assert all(test_x_transformed.columns.values == expected_columns)

    def test_transformations_DummyEncodeTransformer_na_values(self):
        insurance_data = TestHelper.get_insurance_data()

        def test_na_values(data):
            random_row_indexes = data.sample(n=400, random_state=42).index.values
            data.loc[random_row_indexes, 'sex'] = np.nan
            # also set numeric field to NA; values should not change
            data.loc[random_row_indexes, 'bmi'] = np.nan
            assert data.sex.isnull().sum() == 400
            assert data.bmi.isnull().sum() == 400

            expected_columns_one_hot_ignore = ['age', 'bmi', 'children', 'expenses', 'sex_female', 'sex_male',
                                               'smoker_no', 'smoker_yes', 'region_northeast',
                                               'region_northwest', 'region_southeast', 'region_southwest']
            expected_columns_one_hot_not_ignore = ['age', 'bmi', 'children', 'expenses', 'sex_NA',
                                                   'sex_female', 'sex_male', 'smoker_no', 'smoker_yes',
                                                   'region_northeast', 'region_northwest', 'region_southeast',
                                                   'region_southwest']
            expected_columns_dummy_ignore = ['age', 'bmi', 'children', 'expenses', 'sex_male', 'smoker_yes',
                                             'region_northwest', 'region_southeast', 'region_southwest']
            expected_columns_dummy_not_ignore = ['age', 'bmi', 'children', 'expenses', 'sex_female',
                                                 'sex_male', 'smoker_yes', 'region_northwest',
                                                 'region_southeast', 'region_southwest']

            ##################################################################################################
            # ignore_na_values=True (defualt) / ONE_HOT
            ##################################################################################################
            transformation = DummyEncodeTransformer(encoding=CategoricalEncoding.ONE_HOT, ignore_na_values=True)  # noqa
            transformed_data = transformation.fit_transform(data)
            assert transformation._columns_to_reindex == expected_columns_one_hot_ignore
            # ensure all 800 values (i.e. 400 rows by 2 columns) equal 0 (because they are missing values)
            assert (transformed_data.loc[random_row_indexes, ['sex_female', 'sex_male']] == 0).sum().sum() == 800  # noqa
            assert 'sex_NA' not in transformed_data.columns.values
            assert transformed_data.loc[random_row_indexes, 'bmi'].isnull().sum() == 400  # i.e. all still NA

            ##################################################################################################
            # ignore_na_values=True (defualt) / DUMMY
            ##################################################################################################
            transformation = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY, ignore_na_values=True)
            transformed_data = transformation.fit_transform(data)
            assert transformation._columns_to_reindex == expected_columns_dummy_ignore
            # ensure all 400 values of remaining column have 0s
            assert (transformed_data.loc[random_row_indexes, ['sex_male']] == 0).sum().sum() == 400
            assert 'sex_NA' not in transformed_data.columns.values
            assert transformed_data.loc[random_row_indexes, 'bmi'].isnull().sum() == 400  # i.e. all still NA

            ##################################################################################################
            # ignore_na_values=False / ONE_HOT
            ##################################################################################################
            transformation = DummyEncodeTransformer(encoding=CategoricalEncoding.ONE_HOT, ignore_na_values=False)  # noqa
            transformed_data = transformation.fit_transform(data)
            assert transformation._columns_to_reindex == expected_columns_one_hot_not_ignore
            # ensure all 800 values (i.e. 400 rows by 2 columns) equal 0 (because they are missing values)
            assert (transformed_data.loc[random_row_indexes, ['sex_female', 'sex_male']] == 0).sum().sum() == 800  # noqa
            assert (transformed_data.loc[random_row_indexes, ['sex_NA']] == 1).sum().sum() == 400
            assert 'sex_NA' in transformed_data.columns.values
            assert transformed_data.loc[random_row_indexes, 'bmi'].isnull().sum() == 400  # i.e. all still NA

            ##################################################################################################
            # ignore_na_values=False / DUMMY
            ##################################################################################################
            transformation = DummyEncodeTransformer(encoding=CategoricalEncoding.DUMMY, ignore_na_values=False)  # noqa
            transformed_data = transformation.fit_transform(data)
            assert transformation._columns_to_reindex == expected_columns_dummy_not_ignore
            # ensure all 800 values (i.e. 400 rows by 2 columns) equal 0 (because they are missing values)
            assert (transformed_data.loc[random_row_indexes, ['sex_female', 'sex_male']] == 0).sum().sum() == 800  # noqa
            assert 'sex_NA' not in transformed_data.columns.values
            assert transformed_data.loc[random_row_indexes, 'bmi'].isnull().sum() == 400  # i.e. all still NA

        test_na_values(data=insurance_data)

        # fixing a bug where the column is of type categoric, and when we go to add NA, it isn't an expected
        # category, and we would get `ValueError: fill value must be in categories`; solution was to add
        # .cat.add_categories(['NA']) if the column was categoric
        insurance_data_categoric = CategoricConverterTransformer(columns=['sex']).fit_transform(insurance_data)  # noqa
        assert all(insurance_data_categoric.sex.cat.categories.values == ['female', 'male'])
        test_na_values(data=insurance_data_categoric)

    def test_EncodeInteractionEffectsTransformer(self):
        titanic_data = TestHelper.get_titanic_data()
        expected_state = {'expected_column_names': ['Pclass1_Sexfemale',
                                                    'Pclass1_Sexmale',
                                                    'Pclass2_Sexfemale',
                                                    'Pclass2_Sexmale',
                                                    'Pclass3_Sexfemale',
                                                    'Pclass3_Sexmale'],
                          'columns': ['Pclass', 'Sex']}
        expected_encoded_columns = ['Pclass1_Sexfemale', 'Pclass1_Sexmale', 'Pclass2_Sexfemale',
                                    'Pclass2_Sexmale', 'Pclass3_Sexfemale', 'Pclass3_Sexmale']
        unchanged_columns = ['PassengerId', 'Survived', 'Name', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
                             'Cabin', 'Embarked']
        expected_new_columns = unchanged_columns + expected_encoded_columns

        self.assertRaises(AssertionError,
                          lambda: EncodeInteractionEffectsTransformer(columns=None, possible_values=None))

        self.assertRaises(AssertionError,
                          lambda: EncodeInteractionEffectsTransformer(columns=[], possible_values={}))

        self.assertRaises(AssertionError,
                          lambda: EncodeInteractionEffectsTransformer(columns={}, possible_values=None))

        self.assertRaises(AssertionError,
                          lambda: EncodeInteractionEffectsTransformer(columns=None, possible_values=[]))

        # test simple case
        transformer = EncodeInteractionEffectsTransformer(columns=['Pclass', 'Sex'])
        transformed_data = transformer.fit_transform(data_x=titanic_data)

        assert transformer.state == expected_state
        assert all(transformed_data.columns.values == expected_new_columns)
        # ensure that all the rows add up to 1 (i.e. only one column is encoded)
        assert all(transformed_data[expected_encoded_columns].apply(func=sum, axis=1) == 1)
        # temp = titanic_data[['Pclass', 'Sex']]

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_EncodeInteractionEffectsTransformer_normal.pkl'),  # noqa
                                                     expected_dataframe=transformed_data)

        # now test transforming on a subset (i.e. should still give all expected indexes
        transformed_subset = transformer.transform(data_x=titanic_data.iloc[0:1])
        assert len(transformed_subset.columns.values) == len(expected_new_columns)
        assert all(transformed_subset.columns.values == expected_new_columns)
        assert len(transformed_subset) == 1
        assert transformed_subset.loc[0, 'Pclass3_Sexmale'] == 1
        assert all(transformed_subset[expected_encoded_columns].apply(func=sum, axis=1) == 1)

        ######################################################################################################
        # test the transformation if possible_values is used
        ######################################################################################################
        # give unordered values to make sure the columns are sorted
        possible_values = {'Pclass': [3, 1, 2], 'Sex': ['male', 'female']}
        transformer = EncodeInteractionEffectsTransformer(possible_values=possible_values)
        transformed_data = transformer.fit_transform(data_x=titanic_data)

        assert transformer.state == expected_state
        assert all(transformed_data.columns.values == expected_new_columns)
        # ensure that all the rows add up to 1 (i.e. only one column is encoded)
        assert all(transformed_data[expected_encoded_columns].apply(func=sum, axis=1) == 1)
        # temp = titanic_data[['Pclass', 'Sex']]

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory(
            'data/test_Transformers/test_EncodeInteractionEffectsTransformer_normal.pkl'),
            expected_dataframe=transformed_data)
        # now test transforming on a subset (i.e. should still give all expected indexes
        transformed_subset = transformer.transform(data_x=titanic_data.iloc[0:1])
        assert len(transformed_subset.columns.values) == len(expected_new_columns)
        assert all(transformed_subset.columns.values == expected_new_columns)
        assert len(transformed_subset) == 1
        assert transformed_subset.loc[0, 'Pclass3_Sexmale'] == 1
        assert all(transformed_subset[expected_encoded_columns].apply(func=sum, axis=1) == 1)

        ######################################################################################################
        # now `fit` on a dataset that is missing all of the values
        ######################################################################################################
        # give unordered values to make sure the columns are sorted
        possible_values = {'Pclass': [3, 1, 2], 'Sex': ['male', 'female']}
        transformer = EncodeInteractionEffectsTransformer(possible_values=possible_values)
        # only fit on first row
        transformed_data = transformer.fit_transform(data_x=titanic_data.iloc[0:1])

        # all state and transformations should be the same since we passed in possible_values
        assert transformer.state == expected_state
        assert all(transformed_data.columns.values == expected_new_columns)
        # ensure that all the rows add up to 1 (i.e. only one column is encoded)
        assert all(transformed_data[expected_encoded_columns].apply(func=sum, axis=1) == 1)
        # temp = titanic_data[['Pclass', 'Sex']]

        assert len(transformed_data.columns.values) == len(expected_new_columns)
        assert all(transformed_data.columns.values == expected_new_columns)
        assert len(transformed_data) == 1
        assert transformed_data.loc[0, 'Pclass3_Sexmale'] == 1
        assert all(transformed_data[expected_encoded_columns].apply(func=sum, axis=1) == 1)

        # now test transforming on a subset (i.e. should still give all expected indexes
        new_data = titanic_data.iloc[1:2]
        transformed_subset = transformer.transform(data_x=titanic_data.iloc[1:2])

        assert all(transformed_subset[unchanged_columns].iloc[0].values == new_data[unchanged_columns].iloc[0].values)  # noqa

        assert len(transformed_subset.columns.values) == len(expected_new_columns)
        assert all(transformed_subset.columns.values == expected_new_columns)
        assert len(transformed_subset) == 1
        assert transformed_subset.loc[1, 'Pclass1_Sexfemale'] == 1
        assert all(transformed_subset[expected_encoded_columns].apply(func=sum, axis=1) == 1)

    def test_EncodeInteractionEffectsTransformer_unexpected_values(self):
        """
        this test makes sure that any unexpected values raises an assertion error
        (that's what `possible_values` is supposed to solve, so the user should never see new values
        """
        titanic_data = TestHelper.get_titanic_data()

        transformer = EncodeInteractionEffectsTransformer(columns=['Pclass', 'Sex'])
        transformer.fit_transform(data_x=titanic_data)

        # new value, not expected by the `state` of the transformer
        titanic_data.loc[0, 'Pclass'] = 4

        self.assertRaises(AssertionError,
                          lambda: transformer.transform(data_x=titanic_data))

    def test_EncodeNumericNAsTransformer(self):
        data = TestHelper.get_insurance_data()
        # insert missing values for int and float types
        random_row_indexes = data.sample(n=400, random_state=42).index.values
        data.loc[random_row_indexes, 'age'] = np.nan
        # also set numeric field to NA; values should not change
        data.loc[random_row_indexes, 'bmi'] = np.nan
        assert data.age.isnull().sum() == 400
        assert data.bmi.isnull().sum() == 400

        transformer = EncodeNumericNAsTransformer()
        transformed_data = transformer.fit_transform(data)

        assert transformer.state == {'columns': ['age', 'bmi']}
        assert transformer._columns_to_encode == ['age', 'bmi']
        assert transformer._columns_to_reindex == ['age_NA', 'age', 'sex', 'bmi_NA', 'bmi', 'children', 'smoker', 'region', 'expenses']  # noqa

        # ensure no other columns changed
        assert all(data.sex.values == transformed_data.sex.values)
        assert all(data.children.values == transformed_data.children.values)
        assert all(data.smoker.values == transformed_data.smoker.values)
        assert all(data.region.values == transformed_data.region.values)
        assert all(data.expenses.values == transformed_data.expenses.values)

        # ensure non-na values are the same for original column and 0 for new column
        non_missing_index = data.index.isin(random_row_indexes)
        assert all(data[~non_missing_index].age == transformed_data[~non_missing_index].age)
        assert all(data[~non_missing_index].bmi == transformed_data[~non_missing_index].bmi)
        assert all(transformed_data.loc[random_row_indexes, 'age'] == 0)
        assert all(transformed_data.loc[random_row_indexes, 'bmi'] == 0)

        # ensure na values are 1 for new column
        assert all(transformed_data.loc[random_row_indexes, 'age_NA'] == 1)
        assert all(transformed_data.loc[random_row_indexes, 'bmi_NA'] == 1)
        assert all(transformed_data[~non_missing_index].age_NA == 0)
        assert all(transformed_data[~non_missing_index].bmi_NA == 0)

        ######################################################################################################
        # test `columns_to_encode` and `replacement_value`
        ######################################################################################################
        # now, expenses should not have any NA, but we will encode it anyway, so we should get expenses_NA
        # with all 0's and expenses should be the same as before. age/bmi should behave as previous,
        # with the exception of the new replacement_value
        transformer = EncodeNumericNAsTransformer(columns_to_encode=['age', 'bmi', 'expenses'],
                                                  replacement_value=-1)
        transformed_data = transformer.fit_transform(data)

        assert transformer.state == {'columns': ['age', 'bmi', 'expenses']}
        assert transformer._columns_to_encode == ['age', 'bmi', 'expenses']
        assert transformer._columns_to_reindex == ['age_NA', 'age', 'sex', 'bmi_NA', 'bmi', 'children',
                                                   'smoker', 'region', 'expenses_NA', 'expenses']

        # ensure no other columns changed
        assert all(data.sex.values == transformed_data.sex.values)
        assert all(data.children.values == transformed_data.children.values)
        assert all(data.smoker.values == transformed_data.smoker.values)
        assert all(data.region.values == transformed_data.region.values)
        assert all(data.expenses.values == transformed_data.expenses.values)  # expenses should be unchanged

        # ensure non-na values are the same for original column and 0 for new column
        non_missing_index = data.index.isin(random_row_indexes)
        assert all(data[~non_missing_index].age == transformed_data[~non_missing_index].age)
        assert all(data[~non_missing_index].bmi == transformed_data[~non_missing_index].bmi)
        assert all(transformed_data.loc[random_row_indexes, 'age'] == -1)
        assert all(transformed_data.loc[random_row_indexes, 'bmi'] == -1)

        # ensure na values are 1 for new column
        assert all(transformed_data.loc[random_row_indexes, 'age_NA'] == 1)
        assert all(transformed_data.loc[random_row_indexes, 'bmi_NA'] == 1)
        assert all(transformed_data[~non_missing_index].age_NA == 0)
        assert all(transformed_data[~non_missing_index].bmi_NA == 0)

        # expenses_NA should all be 0, because there where no missing values
        assert all(transformed_data.expenses_NA == 0)

    def test_CategoricConverterTransformer(self):
        data = TestHelper.get_titanic_data()
        # transform 'Embarked' so that it is already categoric, and we will try to convert it again, and make
        # sure nothing breaks
        data = CategoricConverterTransformer(columns=['Embarked']).fit_transform(data)

        test_splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.05)
        training_indexes, test_indexes = test_splitter.split(target_values=data.Survived)

        train_data = data.iloc[training_indexes]
        # the test data contains only a subset of the unique values for SibSp and Parch, so we can test that
        # the transformed data contains all categories in the Categorical object
        test_data = data.iloc[test_indexes]

        # Cabin/Embarked are already categoric, shouldn't break anything
        columns_to_convert = ['Pclass', 'SibSp', 'Parch', 'Cabin', 'Embarked']
        self.assertRaises(KeyError, lambda: CategoricConverterTransformer(columns=columns_to_convert+['doesnt exist']).fit(train_data))  # noqa
        transformer = CategoricConverterTransformer(columns=columns_to_convert)
        transformer.fit(train_data)
        assert test_data.Pclass.dtype.name != 'category'
        assert test_data.SibSp.dtype.name != 'category'
        assert test_data.Parch.dtype.name != 'category'
        assert test_data.Cabin.dtype.name == 'object'  # object
        assert test_data.Embarked.dtype.name == 'category'  # already categoric
        new_data = transformer.transform(test_data)
        assert new_data.Pclass.dtype.name == 'category'
        assert new_data.SibSp.dtype.name == 'category'
        assert new_data.Parch.dtype.name == 'category'
        assert new_data.Cabin.dtype.name == 'category'
        assert new_data.Embarked.dtype.name == 'category'  # already categoric

        assert ExploreDataset(new_data).categoric_columns == ['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']  # noqa

        assert all([new_data[x].dtype.name != 'category'
                    for x in new_data.columns.values
                    if x not in columns_to_convert])

        # check state has all correct values
        assert all([transformer.state[x] == sorted(train_data[x].dropna().unique().tolist())
                    for x in columns_to_convert])

        # check Categorical objects have correct values
        assert all([new_data[x].cat.categories.values.tolist() == sorted(train_data[x].dropna().unique().tolist())  # noqa
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
        test_splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.05)
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

        # `fit()` should fail if coluns don't exist
        self.assertRaises(AssertionError,
                          lambda: RemoveColumnsTransformer(columns=['expensess']).fit(data_x=data))

        # test removing each individual columns
        for x in data.columns.values:
            test_remove_columns(columns_to_remove=[x])

        # test removing incrementally more columns
        for index in range(1, len(data.columns.values)):
            test_remove_columns(columns_to_remove=list(data.columns.values[0:(index+1)]))

    def test_BooleanToIntegerTransformer(self):
        zeros_ones = np.random.randint(2, size=100)
        booleans = zeros_ones == 1
        data = pd.DataFrame({'a': booleans,
                             'b': booleans,
                             'c': zeros_ones,
                             'd': [str(val) for val in zeros_ones],
                             'e': [str(val) for val in booleans]})

        assert [OOLearningHelpers.is_series_boolean(data[x]) for x in data.columns.values] == [True, True, False, False, False]  # noqa

        ######################################################################################################
        # all boolean columns (columns parameter is None
        ######################################################################################################
        trans = BooleanToIntegerTransformer()
        transformed_data = trans.fit_transform(data_x=data)
        # check that data didn't change
        assert all(data['a'] == booleans)  # this will be true regardless of conversion, just checking values
        assert OOLearningHelpers.is_series_boolean(data.a)
        assert all(data['b'] == booleans)
        assert OOLearningHelpers.is_series_boolean(data.b)
        assert all(data['c'] == zeros_ones)
        assert not OOLearningHelpers.is_series_boolean(data.c)
        assert all(data['d'] == [str(val) for val in zeros_ones])
        assert not OOLearningHelpers.is_series_boolean(data.d)
        assert all(data['e'] == [str(val) for val in booleans])
        assert not OOLearningHelpers.is_series_boolean(data.e)

        assert all(transformed_data['a'] == zeros_ones)
        assert not OOLearningHelpers.is_series_boolean(transformed_data.a)
        assert all(transformed_data['b'] == zeros_ones)
        assert not OOLearningHelpers.is_series_boolean(transformed_data.b)
        assert all(transformed_data['c'] == zeros_ones)
        assert not OOLearningHelpers.is_series_boolean(transformed_data.c)
        assert all(transformed_data['d'] == [str(val) for val in zeros_ones])
        assert not OOLearningHelpers.is_series_boolean(transformed_data.d)
        assert all(transformed_data['e'] == [str(val) for val in booleans])
        assert not OOLearningHelpers.is_series_boolean(transformed_data.e)

        ######################################################################################################
        # specify columns
        ######################################################################################################
        trans = BooleanToIntegerTransformer(columns=['b'])
        transformed_data = trans.fit_transform(data_x=data)
        # check that data didn't change
        assert all(data['a'] == booleans)  # this will be true regardless of conversion, just checking values
        assert OOLearningHelpers.is_series_boolean(data.a)
        assert all(data['b'] == booleans)
        assert OOLearningHelpers.is_series_boolean(data.b)
        assert all(data['c'] == zeros_ones)
        assert not OOLearningHelpers.is_series_boolean(data.c)
        assert all(data['d'] == [str(val) for val in zeros_ones])
        assert not OOLearningHelpers.is_series_boolean(data.d)
        assert all(data['e'] == [str(val) for val in booleans])
        assert not OOLearningHelpers.is_series_boolean(data.e)

        assert all(transformed_data['a'] == booleans)
        assert OOLearningHelpers.is_series_boolean(transformed_data.a)  # this should be boolean this time
        assert all(transformed_data['b'] == zeros_ones)
        assert not OOLearningHelpers.is_series_boolean(transformed_data.b)
        assert all(transformed_data['c'] == zeros_ones)
        assert not OOLearningHelpers.is_series_boolean(transformed_data.c)
        assert all(transformed_data['d'] == [str(val) for val in zeros_ones])
        assert not OOLearningHelpers.is_series_boolean(transformed_data.d)
        assert all(transformed_data['e'] == [str(val) for val in booleans])
        assert not OOLearningHelpers.is_series_boolean(transformed_data.e)

        # should fail beceause we aren't passing a list
        self.assertRaises(AssertionError, lambda: BooleanToIntegerTransformer(columns='asdf'))
        # should fail beceause column doesn't exist
        self.assertRaises(AssertionError, lambda: BooleanToIntegerTransformer(columns=['f']).fit_transform(data_x=data))  # noqa
        # should fail beceause column isn't a boolean type
        self.assertRaises(AssertionError, lambda: BooleanToIntegerTransformer(columns=['c']).fit_transform(data_x=data))  # noqa

    def test_StatelessTransformer(self):
        """
        Create a StatelessTransformer that does the same thing as RemoveColumnsTransformer so that we can
        duplicate the test and ensure we get the same values. i.e. we should be indifferent towards methods
        """
        data = TestHelper.get_insurance_data()

        def test_remove_columns(columns_to_remove):
            def remove_columns_helper(data_to_transform: pd.DataFrame):
                return data_to_transform.drop(columns=columns_to_remove)

            remove_column_transformer = StatelessTransformer(custom_function=remove_columns_helper)
            new_data = remove_column_transformer.fit_transform(data_x=data)

            assert all(new_data.columns.values ==
                       [column for column in data.columns.values if column not in columns_to_remove])

        self.assertRaises(BaseException, lambda: test_remove_columns(columns_to_remove=['expensess']))

        # test removing each individual columns
        for x in data.columns.values:
            test_remove_columns(columns_to_remove=[x])

        # test removing incrementally more columns
        for index in range(1, len(data.columns.values)):
            test_remove_columns(columns_to_remove=list(data.columns.values[0:(index+1)]))

    def test_StatelessColumnsTransformer(self):
        data = TestHelper.get_insurance_data()
        data.expenses.hist(bins=20)
        data.bmi.hist(bins=20)

        transformer = StatelessColumnTransformer(columns=['expenses', 'bmi'],
                                                 custom_function=lambda x: np.log(x + 1))
        transformed_data = transformer.fit_transform(data_x=data)
        transformed_data.expenses.hist(bins=20)
        transformed_data.bmi.hist(bins=20)

        # check that column order was retained; (even though in our list we reversed above
        assert all(data.columns.values == transformed_data.columns.values)
        # check expected changes
        assert all([x == y for x, y in zip(transformed_data.expenses.values, np.log(data.expenses.values + 1))])  # noqa
        assert all([x == y for x, y in zip(transformed_data.bmi.values, np.log(data.bmi.values + 1))])
        # check that all other columns were not changed
        assert all([x == y for x, y in zip(data.age.values, transformed_data.age.values)])
        assert all([x == y for x, y in zip(data.sex.values, transformed_data.sex.values)])
        assert all([x == y for x, y in zip(data.children.values, transformed_data.children.values)])
        assert all([x == y for x, y in zip(data.smoker.values, transformed_data.smoker.values)])
        assert all([x == y for x, y in zip(data.region.values, transformed_data.region.values)])

        # same thing let's transform with `transform` rather than `fit_transform`
        transformed_data2 = transformer.transform(data)
        # check that column order was retained; (even though in our list we reversed above
        assert all(data.columns.values == transformed_data2.columns.values)
        # check expected changes
        assert all([x == y for x, y in zip(transformed_data2.expenses.values, np.log(data.expenses.values + 1))])  # noqa
        assert all([x == y for x, y in zip(transformed_data2.bmi.values, np.log(data.bmi.values + 1))])
        # check that all other columns were not changed
        assert all([x == y for x, y in zip(data.age.values, transformed_data2.age.values)])
        assert all([x == y for x, y in zip(data.sex.values, transformed_data2.sex.values)])
        assert all([x == y for x, y in zip(data.children.values, transformed_data2.children.values)])
        assert all([x == y for x, y in zip(data.smoker.values, transformed_data2.smoker.values)])
        assert all([x == y for x, y in zip(data.region.values, transformed_data2.region.values)])

    def test_CenterScaleTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        transformer = CenterScaleTransformer()

        # ensure that we are forced to call `fit` first
        self.assertRaises(NotExecutedError, lambda: transformer.transform(data_x=data))

        transformed_training = transformer.fit_transform(data_x=training_set.copy())

        assert isclose(transformer._state['averages']['longitude'], training_set['longitude'].mean())
        assert isclose(transformer._state['averages']['latitude'], training_set['latitude'].mean())
        assert isclose(transformer._state['averages']['housing_median_age'], training_set['housing_median_age'].mean())  # noqa
        assert isclose(transformer._state['averages']['total_rooms'], training_set['total_rooms'].mean())
        assert isclose(transformer._state['averages']['total_bedrooms'], training_set['total_bedrooms'].mean())  # noqa
        assert isclose(transformer._state['averages']['population'], training_set['population'].mean())
        assert isclose(transformer._state['averages']['households'], training_set['households'].mean())
        assert isclose(transformer._state['averages']['median_income'], training_set['median_income'].mean())

        assert isclose(transformer._state['standard_deviations']['longitude'], training_set['longitude'].std(ddof=0))  # noqa
        assert isclose(transformer._state['standard_deviations']['latitude'], training_set['latitude'].std(ddof=0))  # noqa
        assert isclose(transformer._state['standard_deviations']['housing_median_age'], training_set['housing_median_age'].std(ddof=0))  # noqa
        assert isclose(transformer._state['standard_deviations']['total_rooms'], training_set['total_rooms'].std(ddof=0))  # noqa
        assert isclose(transformer._state['standard_deviations']['total_bedrooms'], training_set['total_bedrooms'].std(ddof=0))  # noqa
        assert isclose(transformer._state['standard_deviations']['population'], training_set['population'].std(ddof=0))  # noqa
        assert isclose(transformer._state['standard_deviations']['households'], training_set['households'].std(ddof=0))  # noqa
        assert isclose(transformer._state['standard_deviations']['median_income'], training_set['median_income'].std(ddof=0))  # noqa

        # all transformed columns of the training set should have a mean of zero and a standard deviation of 1
        assert isclose(round(transformed_training['longitude'].mean(), 14), 0)
        assert isclose(round(transformed_training['latitude'].mean(), 14), 0)
        assert isclose(round(transformed_training['housing_median_age'].mean(), 14), 0)
        assert isclose(round(transformed_training['total_rooms'].mean(), 14), 0)
        assert isclose(round(transformed_training['total_bedrooms'].mean(), 14), 0)
        assert isclose(round(transformed_training['population'].mean(), 14), 0)
        assert isclose(round(transformed_training['households'].mean(), 14), 0)
        assert isclose(transformed_training['longitude'].std(ddof=0), 1)
        assert isclose(transformed_training['latitude'].std(ddof=0), 1)
        assert isclose(transformed_training['housing_median_age'].std(ddof=0), 1)
        assert isclose(transformed_training['total_rooms'].std(ddof=0), 1)
        assert isclose(transformed_training['total_bedrooms'].std(ddof=0), 1)
        assert isclose(transformed_training['households'].std(ddof=0), 1)
        assert isclose(transformed_training['population'].std(ddof=0), 1)

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
        assert isclose(round(transformed_test['longitude'].std(ddof=0), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['latitude'].std(ddof=0), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['housing_median_age'].std(ddof=0), 1), 1)
        assert isclose(round(transformed_test['total_rooms'].std(ddof=0), 0), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['total_bedrooms'].std(ddof=0), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['households'].std(ddof=0), 1), 1)  # less precision, but ~1
        assert isclose(round(transformed_test['population'].std(ddof=0), 0), 1)  # less precision, but ~1

    def test_CenterScaleTransformer_column_all_same_values(self):
        """
        the standard deviation of a column containing all of the same number is 0, which creates a problem
        because we divide by the standard deviation (i.e. 0) which gives NaN values.
        in this case, if there is no standard deviation, all of the valuse should simply be 0 for
        Center/Scaling around a mean of 0
        """
        data = TestHelper.get_housing_data()

        data['longitude'] = [1] * len(data)
        assert all(data.longitude.values == 1)
        assert data.drop(columns='total_bedrooms').isna().sum().sum() == 0

        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        transformer = CenterScaleTransformer()
        transformed_training = transformer.fit_transform(data_x=training_set.copy())
        assert transformed_training.drop(columns='total_bedrooms').isna().sum().sum() == 0
        assert all(transformed_training.longitude == 0)

        transformed_test = transformer.transform(data_x=test_set.copy())
        assert transformed_test.drop(columns='total_bedrooms').isna().sum().sum() == 0
        assert all(transformed_test.longitude == 0)

        # introduce variation, should fail since we don't know how tocreate the z-score
        test_set.loc[3396, 'longitude'] = 2
        assert not all(test_set.longitude == 1)
        self.assertRaises(AssertionError, lambda: transformer.transform(data_x=test_set.copy()))

    def test_CenterScaleTransformer_sklearn(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        transformer = CenterScaleTransformer()

        # ensure that we are forced to call `fit` first
        self.assertRaises(NotExecutedError, lambda: transformer.transform(data_x=data))

        transformed_training = transformer.fit_transform(data_x=training_set.copy())

        standard_scaler = StandardScaler()
        sklearn_transformations = standard_scaler.fit_transform(X=training_set[['longitude',
                                                                                'latitude',
                                                                                'housing_median_age',
                                                                                'total_rooms',
                                                                                # 'total_bedrooms',
                                                                                'population',
                                                                                'households',
                                                                                'median_income',
                                                                                ]].copy())
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 0],
                                                  transformed_training.longitude)])

    def test_NormalizationTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        transformer = NormalizationTransformer()

        # ensure that we are forced to call `fit` first
        self.assertRaises(NotExecutedError, lambda: transformer.transform(data_x=data))

        transformed_training = transformer.fit_transform(data_x=training_set.copy())
        assert transformer.state == {'minimums': {'longitude': -124.35, 'latitude': 32.54, 'housing_median_age': 1.0, 'total_rooms': 2.0, 'total_bedrooms': 1.0, 'population': 3.0, 'households': 1.0, 'median_income': 0.4999}, 'maximums': {'longitude': -114.31, 'latitude': 41.95, 'housing_median_age': 52.0, 'total_rooms': 37937.0, 'total_bedrooms': 5471.0, 'population': 16122.0, 'households': 5189.0, 'median_income': 15.0001}}  # noqa

        # categorical columns should not have changed
        assert all(transformed_training.ocean_proximity.values == training_set.ocean_proximity.values)
        assert all(transformed_training.temp_categorical.values == training_set.temp_categorical.values)

        standard_scaler = MinMaxScaler()
        sklearn_transformations = standard_scaler.fit_transform(X=training_set[['longitude',
                                                                                'latitude',
                                                                                'housing_median_age',
                                                                                'total_rooms',
                                                                                # 'total_bedrooms',
                                                                                'population',
                                                                                'households',
                                                                                'median_income',
                                                                                ]].copy())
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 0], transformed_training.longitude)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 1], transformed_training.latitude)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 2], transformed_training.housing_median_age)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 3], transformed_training.total_rooms)])  # noqa
        # assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 0], transformed_training.total_bedrooms)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 4], transformed_training.population)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 5], transformed_training.households)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_transformations[:, 6], transformed_training.median_income)])  # noqa

        # predict
        transformed_test = transformer.transform(data_x=test_set.copy())
        sklearn_test_transformations = standard_scaler.transform(X=test_set[['longitude',
                                                                             'latitude',
                                                                             'housing_median_age',
                                                                             'total_rooms',
                                                                             # 'total_bedrooms',
                                                                             'population',
                                                                             'households',
                                                                             'median_income',
                                                                             ]].copy())
        assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 0], transformed_test.longitude)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 1], transformed_test.latitude)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 2], transformed_test.housing_median_age)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 3], transformed_test.total_rooms)])  # noqa
        # assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 0], transformed_test.total_bedrooms)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 4], transformed_test.population)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 5], transformed_test.households)])  # noqa
        assert all([isclose(x, y) for x, y in zip(sklearn_test_transformations[:, 6], transformed_test.median_income)])  # noqa

    def test_NormalizationVectorSpaceTransformer(self):
        data = TestHelper.get_iris_data()
        transformer = NormalizationVectorSpaceTransformer()
        transformed_data = transformer.fit_transform(data_x=data)

        assert all(transformed_data.index.values == data.index.values)
        assert all(transformed_data.columns.values == data.columns.values)
        assert all(transformed_data.species.values == data.species.values)

        data_normalized = normalize(data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
        assert all([isclose(x, y) for x, y in zip(data_normalized[:, 0], transformed_data.sepal_length)])
        assert all([isclose(x, y) for x, y in zip(data_normalized[:, 1], transformed_data.sepal_width)])
        assert all([isclose(x, y) for x, y in zip(data_normalized[:, 2], transformed_data.petal_length)])
        assert all([isclose(x, y) for x, y in zip(data_normalized[:, 3], transformed_data.petal_width)])

    def test_BoxCoxTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

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

        assert transformer.state == {'housing_median_age': 0.8057453700999767,
                                     'total_rooms': 0.22722010735989,
                                     'total_bedrooms': 8.472135811722177,
                                     'population': 0.24616608260632827,
                                     'households': 0.24939759642245168,
                                     'median_income': 0.08392197805943374}

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

    def test_RemoveNZPTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        remove_nzv_transformer = RemoveNZVTransformer()
        # ensure that we are forced to call `fit` first
        self.assertRaises(NotExecutedError, lambda: remove_nzv_transformer.transform(data_x=training_set))
        remove_nzv_transformer.fit(data_x=training_set)
        assert remove_nzv_transformer.state == {'columns_to_remove': ['longitude']}

        # test with data that
        transformed_data = remove_nzv_transformer.transform(data_x=training_set)
        # verify original data
        assert all(training_set.columns.values == ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'temp_categorical'])  # noqa
        # verify transformed data has expected column removed
        assert all(transformed_data.columns.values == ['latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'temp_categorical'])  # noqa
        assert all(transformed_data.index.values == training_set.index.values)

        self.assertRaises(AlreadyExecutedError, lambda: remove_nzv_transformer.fit(data_x=training_set))

        # test on test set
        # verify original data
        transformed_data = remove_nzv_transformer.transform(data_x=test_set)
        assert all(test_set.columns.values == ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'temp_categorical'])  # noqa
        # verify transformed data has expected column removed
        assert all(transformed_data.columns.values == ['latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'temp_categorical'])  # noqa
        assert all(transformed_data.index.values == test_set.index.values)

        # fit on dataset that doesn't have any NZV
        remove_nzv_transformer2 = RemoveNZVTransformer()
        remove_nzv_transformer2.fit(data_x=transformed_data)
        assert remove_nzv_transformer2.state == {'columns_to_remove': []}
        double_transformed = remove_nzv_transformer2.transform(data_x=transformed_data)
        # ensure both datasets have the same values
        assert all(double_transformed.columns.values == ['latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'ocean_proximity', 'temp_categorical'])  # noqa
        assert all(double_transformed.index.values == transformed_data.index.values)
        assert all(double_transformed == transformed_data)

    def test_PCATransformer(self):
        data = TestHelper.get_housing_data()

        # PCA cannot be used on missing values; data must be imputed or removed
        pca_transformer = PCATransformer()
        self.assertRaises(AssertionError, lambda: pca_transformer.fit(data_x=data))
        data.drop(columns='total_bedrooms', inplace=True)

        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        pca_transformer = PCATransformer()
        # ensure that we are forced to call `fit` first
        self.assertRaises(NotExecutedError, lambda: pca_transformer.transform(data_x=training_set))

        assert pca_transformer.cumulative_explained_variance is None
        assert pca_transformer.number_of_components is None
        assert pca_transformer.state is None

        pca_transformer.fit(data_x=training_set)
        assert all([isclose(x, y) for x, y in zip(pca_transformer.cumulative_explained_variance, [0.955751167228117])])  # noqa
        assert all([isclose(x, y) for x, y in zip(pca_transformer.component_explained_variance, [0.955751167228117])])  # noqa
        assert pca_transformer.number_of_components == 1
        assert pca_transformer.state == {'categorical_features': ['ocean_proximity', 'temp_categorical']}

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_loadings_1.pkl'),  # noqa
                                                     expected_dataframe=pca_transformer.loadings())

        TestHelper.ensure_series_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_loadings_1_top3.pkl'),  # noqa
                                                 expected_series=pca_transformer.loadings(top_n_features=3))  # noqa

        TestHelper.ensure_series_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_comp_feature_ranking_1_all.pkl'),  # noqa
                                                 expected_series=pca_transformer.component_feature_ranking(ith_component=1))  # noqa
        TestHelper.ensure_series_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_comp_feature_ranking_1_1.pkl'),  # noqa
                                                 expected_series=pca_transformer.component_feature_ranking(ith_component=1, top_n=1))  # noqa
        TestHelper.ensure_series_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_comp_feature_ranking_1_4.pkl'),  # noqa
                                                 expected_series=pca_transformer.component_feature_ranking(ith_component=1, top_n=4))  # noqa
        TestHelper.check_plot('data/test_Transformers/test_plot_loadings_1.png',
                              lambda: pca_transformer.plot_loadings())

        # transform training data
        transformed_data = pca_transformer.transform(data_x=training_set)
        assert all(transformed_data.columns.values == ['component_1', 'ocean_proximity', 'temp_categorical'])
        assert all(transformed_data.index.values == training_set.index.values)
        assert all(transformed_data['ocean_proximity'].values == training_set['ocean_proximity'].values)
        assert all(transformed_data['temp_categorical'].values == training_set['temp_categorical'].values)

        # transform test data
        transformed_data = pca_transformer.transform(data_x=test_set)
        assert all(transformed_data.columns.values == ['component_1', 'ocean_proximity', 'temp_categorical'])
        assert all(transformed_data.index.values == test_set.index.values)
        assert all(transformed_data['ocean_proximity'].values == test_set['ocean_proximity'].values)
        assert all(transformed_data['temp_categorical'].values == test_set['temp_categorical'].values)

        # test when setting `percent_variance_explained=None`
        pca_transformer = PCATransformer(percent_variance_explained=None)
        pca_transformer.fit(data_x=training_set)

        assert all([isclose(x, y) for x, y in zip(list(pca_transformer.cumulative_explained_variance), [0.955751167228117, 0.9976620662281546, 0.9999751439740711, 0.9999981341821631, 0.9999994868435985, 0.9999999583168993, 1.0])])  # noqa
        assert all([isclose(x, y) for x, y in zip(list(pca_transformer.component_explained_variance), [0.955751167228117, 0.04191089900003746, 0.002313077745916491, 2.299020809206057e-05, 1.3526614353928983e-06, 4.714733007559294e-07, 4.16831008376836e-08])])  # noqa
        assert pca_transformer.number_of_components == 7
        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_loadings_7.pkl'),  # noqa
                                                     expected_dataframe=pca_transformer.loadings())
        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_loadings_7_top4cats_top3feats.pkl'),  # noqa
                                                     expected_dataframe=pca_transformer.loadings(top_n_components=4, top_n_features=3))  # noqa

        TestHelper.ensure_series_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_comp_feature_ranking_1_all.pkl'),  # noqa
                                                 expected_series=pca_transformer.component_feature_ranking(ith_component=1))  # noqa
        TestHelper.ensure_series_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_comp_feature_ranking_7_2.pkl'),  # noqa
                                                 expected_series=pca_transformer.component_feature_ranking(ith_component=2))  # noqa
        TestHelper.ensure_series_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_PCATransformer_comp_feature_ranking_7_3_4.pkl'),  # noqa
                                                 expected_series=pca_transformer.component_feature_ranking(ith_component=6, top_n=4))  # noqa

        TestHelper.check_plot('data/test_Transformers/test_plot_loadings_7_filter.png',
                              lambda: pca_transformer.plot_loadings(top_n_components=4, top_n_features=3))
        TestHelper.check_plot('data/test_Transformers/test_plot_loadings_7.png',
                              lambda: pca_transformer.plot_loadings(font_size=5))
        TestHelper.check_plot('data/test_Transformers/test_plot_loadings_7_no_annotate.png',
                              lambda: pca_transformer.plot_loadings(annotate=False))

        assert pca_transformer.state == {'categorical_features': ['ocean_proximity', 'temp_categorical']}

        # transform training data
        transformed_data = pca_transformer.transform(data_x=training_set)
        assert all(transformed_data.columns.values == ['component_1', 'component_2', 'component_3', 'component_4', 'component_5', 'component_6', 'component_7', 'ocean_proximity', 'temp_categorical'])  # noqa
        assert all(transformed_data.index.values == training_set.index.values)
        assert all(transformed_data['ocean_proximity'].values == training_set['ocean_proximity'].values)
        assert all(transformed_data['temp_categorical'].values == training_set['temp_categorical'].values)

        TestHelper.check_plot('data/test_Transformers/test_get_pca_plot.png',
                              lambda: pca_transformer.plot_cumulative_variance())

        # test when setting `exclude_categorical_columns=True`
        pca_transformer = PCATransformer(exclude_categorical_columns=True)
        transformed_data = pca_transformer.fit_transform(data_x=training_set)
        assert all(transformed_data.columns.values == ['component_1'])
        assert all(transformed_data.index.values == training_set.index.values)

        # transform test data
        transformed_data = pca_transformer.transform(data_x=test_set)
        assert all(transformed_data.columns.values == ['component_1'])
        assert all(transformed_data.index.values == test_set.index.values)

    def test_RemoveCorrelationsTransformer(self):
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        training_set, _, test_set, _ = TestHelper.split_train_holdout_regression(data, target_variable)

        # at a threshold of 0.99, nothing should be removed
        expected_removed = []
        transformer = RemoveCorrelationsTransformer(max_correlation_threshold=0.99)
        transformed_data = transformer.fit_transform(data_x=training_set)
        assert transformer.state['columns_to_remove'] == expected_removed
        assert list(transformed_data.columns.values) == [x for x in training_set.columns.values
                                                         if x not in expected_removed]
        transformed_data = transformer.transform(data_x=test_set)
        assert list(transformed_data.columns.values) == [x for x in training_set.columns.values
                                                         if x not in expected_removed]

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

    def test_encode_dates(self):

        # test that the original dataframe is unchanged
        # test on dataframe that doesn't have any date columns

        date_range_1 = pd.date_range(start='1/1/2018', end='1/1/2019', freq='D')
        date_range_2 = pd.date_range(start='1/1/2017', end='1/1/2018', freq='D')
        assert len(date_range_1) == len(date_range_2)
        date_dataframe = pd.DataFrame({'id': range(len(date_range_1)),
                                       'dates_1': date_range_1,
                                       'temp1': [1] * len(date_range_1),
                                       'dates_2': date_range_2,
                                       'temp2': [2] * len(date_range_1)})

        transformed_data = EncodeDateColumnsTransformer().fit_transform(data_x=date_dataframe)

        expected_columns = ['id', 'temp1', 'temp2', 'dates_1_year', 'dates_1_month', 'dates_1_day',
                            'dates_1_hour', 'dates_1_minute', 'dates_1_second', 'dates_1_quarter',
                            'dates_1_week', 'dates_1_days_in_month', 'dates_1_day_of_year',
                            'dates_1_day_of_week', 'dates_1_is_leap_year', 'dates_1_is_month_end',
                            'dates_1_is_month_start', 'dates_1_is_quarter_end', 'dates_1_is_quarter_start',
                            'dates_1_is_year_end', 'dates_1_is_year_start', 'dates_1_is_us_federal_holiday',
                            'dates_1_is_weekday', 'dates_1_is_weekend', 'dates_2_year', 'dates_2_month',
                            'dates_2_day', 'dates_2_hour', 'dates_2_minute', 'dates_2_second',
                            'dates_2_quarter', 'dates_2_week', 'dates_2_days_in_month', 'dates_2_day_of_year',
                            'dates_2_day_of_week', 'dates_2_is_leap_year', 'dates_2_is_month_end',
                            'dates_2_is_month_start', 'dates_2_is_quarter_end', 'dates_2_is_quarter_start',
                            'dates_2_is_year_end', 'dates_2_is_year_start', 'dates_2_is_us_federal_holiday',
                            'dates_2_is_weekday', 'dates_2_is_weekend']

        assert all(transformed_data.columns.values == expected_columns)
        # ensure original didn't change
        assert all(date_dataframe.columns.values == ['id', 'dates_1', 'temp1', 'dates_2', 'temp2'])

        TestHelper.ensure_all_values_equal_from_file(file=TestHelper.ensure_test_directory('data/test_Transformers/test_EncodeDateColumnsTransformer.pkl'),  # noqa
                                                     expected_dataframe=transformed_data)

        ######################################################################################################
        # test `include_columns`
        ######################################################################################################
        include_columns = EncodeDateColumnsTransformer.encoded_columns()[0:5]
        self.assertRaises(AssertionError,
                          lambda: EncodeDateColumnsTransformer(include_columns=include_columns+['fail']))

        transformed_data_subset = EncodeDateColumnsTransformer(include_columns=include_columns).\
            fit_transform(data_x=date_dataframe)

        assert all(transformed_data_subset.columns.values == ['id', 'temp1', 'temp2', 'dates_1_year', 'dates_1_month', 'dates_1_day', 'dates_1_hour', 'dates_1_minute', 'dates_2_year', 'dates_2_month', 'dates_2_day', 'dates_2_hour', 'dates_2_minute'])  # noqa

        for column in transformed_data_subset.columns.values:
            assert all(transformed_data_subset[column].values == transformed_data[column].values)
