import os
import pickle
import matplotlib.pyplot as plt

import pandas as pd

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


class MockExploreBase(ExploreDatasetBase):
    """
    only used to instantiate an abstract class
    """
    pass


# noinspection PyMethodMayBeStatic
class ExploratoryTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_OOLearningHelpers(self):
        data = pd.read_csv(TestHelper.ensure_test_directory('data/credit.csv'))
        numeric_features, categoric_features = OOLearningHelpers.get_columns_by_type(data_dtypes=data.dtypes)
        assert numeric_features == ['months_loan_duration', 'amount', 'percent_of_income', 'years_at_residence', 'age', 'existing_loans_count', 'dependents']  # noqa
        assert categoric_features == ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_duration', 'other_credit', 'housing', 'job', 'phone', 'default']  # noqa

        assert OOLearningHelpers.is_series_dtype_numeric(data.months_loan_duration.dtype) is True
        assert OOLearningHelpers.is_series_dtype_numeric(data.amount.dtype) is True
        assert OOLearningHelpers.is_series_dtype_numeric(data.percent_of_income.dtype) is True
        assert OOLearningHelpers.is_series_dtype_numeric(data.years_at_residence.dtype) is True
        assert OOLearningHelpers.is_series_dtype_numeric(data.age.dtype) is True
        assert OOLearningHelpers.is_series_dtype_numeric(data.existing_loans_count.dtype) is True
        assert OOLearningHelpers.is_series_dtype_numeric(data.dependents.dtype) is True

        assert OOLearningHelpers.is_series_dtype_numeric(data.checking_balance.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.credit_history.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.purpose.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.savings_balance.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.employment_duration.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.other_credit.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.housing.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.job.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.phone.dtype) is False
        assert OOLearningHelpers.is_series_dtype_numeric(data.default.dtype) is False

        assert OOLearningHelpers.is_series_numeric(data['months_loan_duration']) is True
        assert OOLearningHelpers.is_series_numeric(data.months_loan_duration) is True
        assert OOLearningHelpers.is_series_numeric(data.amount) is True
        assert OOLearningHelpers.is_series_numeric(data.percent_of_income) is True
        assert OOLearningHelpers.is_series_numeric(data.years_at_residence) is True
        assert OOLearningHelpers.is_series_numeric(data.age) is True
        assert OOLearningHelpers.is_series_numeric(data.existing_loans_count) is True
        assert OOLearningHelpers.is_series_numeric(data.dependents) is True

        assert OOLearningHelpers.is_series_numeric(data['checking_balance']) is False
        assert OOLearningHelpers.is_series_numeric(data.checking_balance) is False
        assert OOLearningHelpers.is_series_numeric(data.credit_history) is False
        assert OOLearningHelpers.is_series_numeric(data.purpose) is False
        assert OOLearningHelpers.is_series_numeric(data.savings_balance) is False
        assert OOLearningHelpers.is_series_numeric(data.employment_duration) is False
        assert OOLearningHelpers.is_series_numeric(data.other_credit) is False
        assert OOLearningHelpers.is_series_numeric(data.housing) is False
        assert OOLearningHelpers.is_series_numeric(data.job) is False
        assert OOLearningHelpers.is_series_numeric(data.phone) is False
        assert OOLearningHelpers.is_series_numeric(data.default) is False

    def test_ExploreDatasetBase_Categories(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        categoric_columns = ['checking_balance', 'credit_history', 'purpose', 'savings_balance',
                             'employment_duration', 'other_credit', 'housing', 'job', 'phone']
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)
        # check all categoric columns are pandas categories
        for feature in categoric_columns:
            assert explore.dataset[feature].dtype.name == 'category'
        # check all non-categoric columns are not pandas categories
        for feature in [x for x in explore.dataset.columns if x not in categoric_columns]:
            assert explore.dataset[feature].dtype.name != 'category'

    def test_ExploreDatasetBase_summary(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        numeric_columns = ['months_loan_duration', 'amount', 'percent_of_income', 'years_at_residence', 'age', 'existing_loans_count', 'dependents']  # noqa
        categoric_columns = ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_duration', 'other_credit', 'housing', 'job', 'phone']  # noqa
        target_variable = 'default'

        explore_from_csv = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)
        assert explore_from_csv is not None
        assert isinstance(explore_from_csv.dataset, pd.DataFrame)
        assert len(explore_from_csv.dataset) == 1000
        assert explore_from_csv.numeric_features == ['months_loan_duration', 'amount', 'percent_of_income', 'years_at_residence', 'age', 'existing_loans_count', 'dependents']  # noqa
        assert explore_from_csv.categoric_features == ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_duration', 'other_credit', 'housing', 'job', 'phone']  # noqa

        explore = MockExploreBase(dataset=pd.read_csv(credit_csv), target_variable=target_variable)
        assert explore is not None
        assert isinstance(explore.dataset, pd.DataFrame)
        assert len(explore.dataset) == 1000

        assert explore.numeric_features == numeric_columns
        assert explore.categoric_features == categoric_columns

        assert TestHelper.ensure_all_values_equal(data_frame1=explore_from_csv.dataset, data_frame2=explore.dataset)  # noqa
        assert explore_from_csv.target_variable == explore.target_variable

        ######################################################################################################
        # numeric
        ######################################################################################################
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Exploratory/test_ExploreDatasetBase_before_mod_numeric.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(explore.numeric_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_summary,
                                                      data_frame2=explore.numeric_summary)

        # introduce None's and Zeros into a couple of variable
        explore.dataset.loc[0, 'months_loan_duration'] = None
        explore.dataset.loc[5, 'months_loan_duration'] = None
        explore.dataset.loc[60, 'months_loan_duration'] = None
        explore.dataset.loc[75, 'months_loan_duration'] = None

        explore.dataset.loc[1, 'percent_of_income'] = None
        explore.dataset.loc[9, 'percent_of_income'] = None
        explore.dataset.loc[68, 'percent_of_income'] = None

        explore.dataset.loc[1, 'months_loan_duration'] = 0
        explore.dataset.loc[6, 'months_loan_duration'] = 0
        explore.dataset.loc[61, 'months_loan_duration'] = 0
        explore.dataset.loc[76, 'months_loan_duration'] = 0
        explore.dataset.loc[77, 'months_loan_duration'] = 0

        explore.dataset.loc[10, 'percent_of_income'] = 0
        explore.dataset.loc[69, 'percent_of_income'] = 0

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Exploratory/test_ExploreDatasetBase_after_mod_numeric.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(explore.numeric_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_summary,
                                                      data_frame2=explore.numeric_summary)

        ######################################################################################################
        # categoric
        ######################################################################################################
        explore = MockExploreBase(dataset=pd.read_csv(credit_csv), target_variable=target_variable)
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Exploratory/test_ExploreDatasetBase_before_mod_categoric.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(explore.categoric_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_summary,
                                                      data_frame2=explore.categoric_summary)
            # ensure that changing the columns to Categorical (when loading from csv) doesn't affect anything
            # make sure at least 1 column is a category
            assert explore_from_csv.dataset['job'].dtype.name == 'category'
            assert explore_from_csv.dataset['age'].dtype.name != 'category'

            assert TestHelper.ensure_all_values_equal(data_frame1=expected_summary,
                                                      data_frame2=explore_from_csv.categoric_summary)

        # introduce None's and Zeros into a couple of variable
        explore.dataset.loc[0, 'checking_balance'] = None
        explore.dataset.loc[5, 'checking_balance'] = None
        explore.dataset.loc[60, 'checking_balance'] = None
        explore.dataset.loc[75, 'checking_balance'] = None

        explore.dataset.loc[1, 'savings_balance'] = None
        explore.dataset.loc[9, 'savings_balance'] = None
        explore.dataset.loc[68, 'savings_balance'] = None

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Exploratory/test_ExploreDatasetBase_after_mod_categoric.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(explore.categoric_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_summary,
                                                      data_frame2=explore.categoric_summary)

    def test_ExploreDatasetBase_unique_values(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)
        # cannot get unique values on numeric feature
        self.assertRaises(AssertionError, lambda: explore.unique_values(categoric_feature='amount'))
        self.assertRaises(AssertionError, lambda: explore.plot_unique_values(categoric_feature='amount'))

        ######################################################################################################
        # `sort_by_features=False`
        ######################################################################################################
        unique_values = explore.unique_values('checking_balance')
        assert sorted(unique_values.index.values) == sorted(explore.dataset.checking_balance.unique())
        assert all(unique_values.freq.values == [394, 274, 269, 63])
        assert all(unique_values.perc.values == [0.394, 0.274, 0.269, 0.063])

        # ensure `unique_values()` also works for the target variable since it is categoric as well
        unique_values = explore.unique_values(target_variable)
        assert sorted(unique_values.index.values) == sorted(explore.dataset[target_variable].unique())
        assert all(unique_values.freq.values == [700, 300])
        assert all(unique_values.perc.values == [0.7, 0.3])

        TestHelper.check_plot('data/test_exploratory/unique_purpose.png',  # noqa
                              lambda: explore.plot_unique_values(categoric_feature='purpose'))

        TestHelper.check_plot('data/test_exploratory/unique_default.png',  # noqa
                              lambda: explore.plot_unique_values(categoric_feature=target_variable))

        ######################################################################################################
        # `sort_by_features=True`
        ######################################################################################################
        # from_csv SETS COLUMNS TO CATEGORICAL
        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)
        unique_values = explore.unique_values('checking_balance', sort_by_feature=True)
        assert all(unique_values.index.values == ['1 - 200 DM', '< 0 DM', '> 200 DM', 'unknown'])
        assert all(unique_values.freq.values == [269, 274, 63, 394])
        assert all(unique_values.perc.values == [0.269, 0.274, 0.063, 0.394])

        # now set the order of categorical
        explore.set_level_order(categoric_feature='checking_balance',
                                levels=['< 0 DM', '1 - 200 DM', '> 200 DM', 'unknown'])
        unique_values = explore.unique_values('checking_balance', sort_by_feature=True)
        assert all(unique_values.index.values == ['< 0 DM', '1 - 200 DM', '> 200 DM', 'unknown'])
        assert all(unique_values.freq.values == [274, 269, 63, 394])
        assert all(unique_values.perc.values == [0.274, 0.269, 0.063, 0.394])

        # not ordered by feature, ordered by frequency
        TestHelper.check_plot('data/test_exploratory/unique_checking_balance_not_sorted.png',
                              lambda: explore.plot_unique_values(categoric_feature='checking_balance',
                                                                 sort_by_feature=False))

        # ordered
        TestHelper.check_plot('data/test_exploratory/unique_checking_balance_sort.png',
                              lambda: explore.plot_unique_values(categoric_feature='checking_balance',
                                                                 sort_by_feature=True))

    # noinspection SpellCheckingInspection
    def test_ExploreDatasetBase_set_as_categoric(self):
        titanic_csv = TestHelper.ensure_test_directory('data/titanic.csv')
        target_variable = 'Survived'

        explore = MockExploreBase.from_csv(csv_file_path=titanic_csv, target_variable=target_variable)
        # we want Pclass to be a categoric feature but it is currently numeric

        feature = 'Pclass'
        assert feature in explore.numeric_features
        assert feature not in explore.categoric_features
        assert all(explore.dataset.iloc[0:10][feature] == [3, 1, 3, 1, 3, 3, 1, 3, 3, 2])
        target_mapping = {1: 'a',
                          2: 'b',
                          3: 'c'}
        explore.set_as_categoric(feature=feature, mapping=target_mapping)
        assert feature in explore.categoric_features
        assert feature not in explore.numeric_features
        # make sure we mapped right values
        assert all(explore.dataset[feature].values.categories.values == list(target_mapping.values()))
        # spot check first 10
        assert all(explore.dataset.iloc[0:10][feature] == ['c', 'a', 'c', 'a', 'c', 'c', 'a', 'c', 'c', 'b'])

    def test_ExploreDatasetBase_histogram(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)

        # cannot get unique values on numeric feature
        self.assertRaises(AssertionError, lambda: explore.plot_histogram(numeric_feature=target_variable))

        TestHelper.check_plot('data/test_exploratory/hist_amount.png',
                              lambda: explore.plot_histogram(numeric_feature='amount'))

        TestHelper.check_plot('data/test_exploratory/hist_years_at_residence.png',
                              lambda: explore.plot_histogram(numeric_feature='years_at_residence'))

    def test_ExploreDatasetBase_boxplot(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)

        # cannot get unique values on numeric feature
        self.assertRaises(AssertionError, lambda: explore.plot_boxplot(numeric_feature=target_variable))

        TestHelper.check_plot('data/test_exploratory/boxplot_amount.png',
                              lambda: explore.plot_boxplot(numeric_feature='amount'))

        TestHelper.check_plot('data/test_exploratory/boxplot_years_at_residence.png',
                              lambda: explore.plot_boxplot(numeric_feature='years_at_residence'))

    def test_ExploreDatasetBase_scatter_plot_numerics(self):
        credit_csv = TestHelper.ensure_test_directory('data/housing.csv')
        target_variable = 'median_house_value'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)

        TestHelper.check_plot('data/test_exploratory/scatter_plot_numerics_subset.png',
                              lambda: explore.plot_scatterplot_numerics(numeric_columns=['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']))  # noqa

    def test_ExploreDatasetBase_correlations(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/credit_correlation_heatmap.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.plot_correlation_heatmap()
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

    def test_ExploreClassificationDataset(self):
        self.assertRaises(ValueError, lambda: ExploreClassificationDataset.from_csv(csv_file_path=TestHelper.ensure_test_directory('data/housing.csv'), target_variable='median_house_value'))  # noqa

        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = ExploreClassificationDataset.from_csv(csv_file_path=credit_csv, target_variable=target_variable)  # noqa

        assert isinstance(explore, ExploreClassificationDataset)
        assert explore.target_variable == target_variable
        assert explore.numeric_features == ['months_loan_duration', 'amount', 'percent_of_income', 'years_at_residence', 'age', 'existing_loans_count', 'dependents']  # noqa
        assert explore.categoric_features == ['checking_balance', 'credit_history', 'purpose', 'savings_balance', 'employment_duration', 'other_credit', 'housing', 'job', 'phone']  # noqa

    def test_ExploreClassificationDataset_numeric_target(self):
        titanic_csv = TestHelper.ensure_test_directory('data/titanic.csv')
        target_variable = 'Survived'
        target_mapping = {0: 'died', 1: 'survived'}

        explore = ExploreDatasetBase.from_csv(csv_file_path=titanic_csv, target_variable=target_variable)
        assert explore._is_target_numeric  # target is numeric, but this could fuck with this
        numeric_data = explore.dataset[target_variable]
        expected_categoric_data = numeric_data.map(target_mapping).values

        explore = ExploreClassificationDataset.from_csv(csv_file_path=titanic_csv,
                                                        target_variable=target_variable,
                                                        map_numeric_target=target_mapping)
        assert explore._is_target_numeric is False
        # noinspection PyTypeChecker
        assert all(explore.dataset[target_variable] == expected_categoric_data)

    # noinspection PyUnresolvedReferences
    def test_ExploreClassificationDataset_categorical_vs_target(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = ExploreClassificationDataset.from_csv(csv_file_path=credit_csv, target_variable=target_variable)  # noqa
        # make sure setting this changes/sorts the order of the bars in the graph (it does)
        explore.set_level_order(categoric_feature='checking_balance',
                                levels=['< 0 DM', '1 - 200 DM', '> 200 DM', 'unknown'])

        # cannot get unique values on numeric feature
        self.assertRaises(AssertionError, lambda: explore.plot_against_target(feature=target_variable))

        TestHelper.check_plot('data/test_exploratory/compare_against_target_checking_balance.png',
                              lambda: explore.plot_against_target(feature='checking_balance'))

        TestHelper.check_plot('data/test_exploratory/compare_against_target_amount.png',
                              lambda: explore.plot_against_target(feature='amount'))

        TestHelper.check_plot('data/test_exploratory/plot_histogram_against_target_amount.png',
                              lambda: explore.plot_histogram_against_target(numeric_feature='amount'))

    def test_ExploreRegressionDataset(self):
        self.assertRaises(ValueError, lambda: ExploreRegressionDataset.from_csv(csv_file_path=TestHelper.ensure_test_directory('data/credit.csv'), target_variable='default'))  # noqa

        housing_csv = TestHelper.ensure_test_directory('data/housing.csv')
        target_variable = 'median_house_value'

        explore = ExploreRegressionDataset.from_csv(csv_file_path=housing_csv, target_variable=target_variable)  # noqa

        assert isinstance(explore, ExploreRegressionDataset)
        assert explore.target_variable == target_variable
        assert explore.numeric_features == ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']  # noqa
        assert explore.categoric_features == ['ocean_proximity']

    # noinspection PyUnresolvedReferences
    def test_ExploreRegressionDataset_categorical_vs_target(self):
        credit_csv = TestHelper.ensure_test_directory('data/housing.csv')
        target_variable = 'median_house_value'

        explore = ExploreRegressionDataset.from_csv(csv_file_path=credit_csv, target_variable=target_variable)  # noqa
        assert isinstance(explore, ExploreRegressionDataset)
        # make sure setting this changes/sorts the order of the bars in the graph (it does)
        explore.set_level_order(categoric_feature='ocean_proximity',
                                levels=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

        # cannot get unique values on numeric feature
        self.assertRaises(AssertionError, lambda: explore.plot_against_target(feature=target_variable))

        TestHelper.check_plot('data/test_exploratory/compare_against_target_ocean_proximity.png',
                              lambda: explore.plot_against_target(feature='ocean_proximity'))

        TestHelper.check_plot('data/test_exploratory/compare_against_target_median_income.png',
                              lambda: explore.plot_against_target(feature='median_income'))
