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
                             'employment_duration', 'other_credit', 'housing', 'job', 'phone']  # noqa
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

        explore = MockExploreBase(dataset=pd.read_csv(credit_csv), target_variable=target_variable)  # noqa
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
        explore = MockExploreBase(dataset=pd.read_csv(credit_csv), target_variable=target_variable)  # noqa
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

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory(
            'data/test_Exploratory/test_ExploreDatasetBase_after_mod_categoric.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(explore.categoric_summary, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_summary = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_summary,
                                                      data_frame2=explore.categoric_summary)

    def test_ExploreDatasetBase_unique_values(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)  # noqa
        # cannot get unique values on numeric feature
        self.assertRaises(AssertionError, lambda: explore.unique_values(categoric_feature='amount'))
        self.assertRaises(AssertionError, lambda: explore.unique_values_bar(categoric_feature='amount'))

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

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/unique_purpose.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.unique_values_bar(categoric_feature='purpose')
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/unique_default.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.unique_values_bar(categoric_feature=target_variable)
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

        ######################################################################################################
        # `sort_by_features=True`
        ######################################################################################################
        # from_csv SETS COLUMNS TO CATEGORICAL
        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)  # noqa
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
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/unique_checking_balance_not_sorted.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.unique_values_bar(categoric_feature='checking_balance', sort_by_feature=False)
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

        # ordered
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/unique_checking_balance_sort.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.unique_values_bar(categoric_feature='checking_balance', sort_by_feature=True)
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

    def test_ExploreDatasetBase_histogram(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)

        # cannot get unique values on numeric feature
        self.assertRaises(AssertionError, lambda: explore.histogram(numeric_feature=target_variable))

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/hist_amount.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.histogram(numeric_feature='amount')
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/hist_years_at_residence.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.histogram(numeric_feature='years_at_residence')
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

    def test_ExploreDatasetBase_correlations(self):
        credit_csv = TestHelper.ensure_test_directory('data/credit.csv')
        target_variable = 'default'

        explore = MockExploreBase.from_csv(csv_file_path=credit_csv, target_variable=target_variable)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_exploratory/credit_correlation_heatmap.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        explore.correlation_heatmap()
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)
