import math
import os

from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection SpellCheckingInspection
class ModelWrapperTests(TimerTestCase):
    """
    Testing the TestHelper might seem like overkill; but what if, for example, there was a small bug that
        didn't report errors in certain cases. Hundreds of hours will have been spent writing unit tests that
        won't get excited because of a bug in a helper method.
    """

    def test_ensure_all_values_equal(self):
        titanic_data = TestHelper.get_titanic_data()
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_TestHelpers/test_ensure_all_values_equal_titanic.pkl'))  # noqa

        # ensure if the file does not exist we get FileNotFoundError
        self.assertRaises(FileNotFoundError, lambda: TestHelper.ensure_all_values_equal_from_file(file=file + 's',  # noqa
                                                                                                  expected_dataframe=titanic_data))  # noqa

        assert math.isnan(titanic_data.loc[5, 'Age'])  # ensure we will test NaN value, for numeric column
        assert math.isnan(titanic_data.loc[0, 'Cabin'])  # ensure we will test NaN value, for categoric column
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=titanic_data)

        # change a numeric value, slightly, and retest, should fail
        titanic_data.loc[0, 'Age'] = titanic_data.loc[0, 'Age'] + 0.000001
        self.assertRaises(AssertionError, lambda: TestHelper.ensure_all_values_equal_from_file(file=file,
                                                                                               expected_dataframe=titanic_data))  # noqa
        # test changing a categoric value
        titanic_data = TestHelper.get_titanic_data()
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=titanic_data)
        titanic_data.loc[0, 'Sex'] = titanic_data.loc[0, 'Sex'] + 's'
        self.assertRaises(AssertionError, lambda: TestHelper.ensure_all_values_equal_from_file(file=file,
                                                                                               expected_dataframe=titanic_data))  # noqa
        # test changing one of the column names
        titanic_data = TestHelper.get_titanic_data()
        columns = titanic_data.columns.values
        new_columns = dict(zip(columns, ['passengerid'] + list(columns[1:])))
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=titanic_data)
        titanic_data.rename(columns=new_columns, inplace=True)
        self.assertRaises(AssertionError, lambda: TestHelper.ensure_all_values_equal_from_file(file=file,
                                                                                               expected_dataframe=titanic_data))  # noqa
        # test changing the index values
        titanic_data = TestHelper.get_titanic_data()
        index = titanic_data.index.values
        new_index = dict(zip(index, ['passengerid'] + list(index[1:])))
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=titanic_data)
        titanic_data.rename(index=new_index, inplace=True)
        self.assertRaises(AssertionError, lambda: TestHelper.ensure_all_values_equal_from_file(file=file,
                                                                                               expected_dataframe=titanic_data))  # noqa
        # check that if we have the same values in a given column, but a different column type (e.g.
        # one DataFrame, for a given column, is numeric, and the other, which has the same 'values' is
        # 'object'
        titanic_data = TestHelper.get_titanic_data()
        titanic_data.Pclass = titanic_data.Pclass.astype(object)
        self.assertRaises(AssertionError, lambda: TestHelper.ensure_all_values_equal_from_file(file=file,
                                                                                               expected_dataframe=titanic_data))  # noqa



