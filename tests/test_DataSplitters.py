import os
import pickle
from math import isclose

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection SpellCheckingInspection,PyMethodMayBeStatic
class DataSplittersTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_splitters_RegressionStratifiedDataSplitter(self):
        holdout_ratio = 0.20
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'

        test_splitter = RegressionStratifiedDataSplitter(holdout_ratio=holdout_ratio)
        training_indexes, test_indexes = test_splitter.split(target_values=data[target_variable])

        assert isinstance(training_indexes, list)
        assert isinstance(test_indexes, list)

        # ensure sets are correct sizes and don't overlap
        assert len(training_indexes) == len(data) * 0.8
        assert len(test_indexes) == len(data) * 0.2
        assert set(training_indexes).isdisjoint(test_indexes)  # no overlapping indexes in training/test

        indexes_file_path = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_DataSplitters/RegressionStratifiedDataSplitter_indexes.pkl'))  # noqa
        # with open(indexes_file_path, 'wb') as output:
        #      pickle.dump(training_indexes+test_indexes, output, pickle.HIGHEST_PROTOCOL)
        with open(indexes_file_path, 'rb') as saved_object:
            training_test_indexes = pickle.load(saved_object)
            assert training_indexes + test_indexes == training_test_indexes

        # visualize distribution of the target variable to visually confirm stratification
        TestHelper.check_plot('data/test_DataSplitters/test_splitters_RegressionStra_distribution_training.png',  # noqa
                              lambda: data.iloc[training_indexes][target_variable].hist(color='blue', edgecolor='black', grid=None))  # noqa

        TestHelper.check_plot('data/test_DataSplitters/test_splitters_RegressionStratifi_distribution_test.png',  # noqa
                              lambda: data.iloc[test_indexes][target_variable].hist(color='blue', edgecolor='black', grid=None))  # noqa

    def test_splitters_ClassificationStratifiedDataSplitter(self):
        holdout_ratio = 0.20
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'

        test_splitter = ClassificationStratifiedDataSplitter(holdout_ratio=holdout_ratio)
        training_indexes, test_indexes = test_splitter.split(target_values=data[target_variable])

        assert len(training_indexes) - len(data) * 0.8 < 2  # needs to be very close
        assert len(test_indexes) - len(data) * 0.2 < 2  # needs to be very close
        assert set(training_indexes).isdisjoint(test_indexes)  # no overlapping indexes in training/test

        indexes_file_path = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_DataSplitters/ClassificationStratifiedDataSplitter_indexes.pkl'))  # noqa
        # with open(indexes_file_path, 'wb') as output:
        #     pickle.dump(training_indexes+test_indexes, output, pickle.HIGHEST_PROTOCOL)
        with open(indexes_file_path, 'rb') as saved_object:
            training_test_indexes = pickle.load(saved_object)
            assert training_indexes + test_indexes == training_test_indexes

        data_target_proportions = data.Survived.value_counts(normalize=True)
        assert all([isclose(x, y) for x, y in zip(data_target_proportions, [0.616161616161, 0.383838383838])])
        training_target_proportions = data.iloc[training_indexes].Survived.value_counts(normalize=True)
        test_target_proportions = data.iloc[test_indexes].Survived.value_counts(normalize=True)
        assert all(abs(data_target_proportions - training_target_proportions) < 0.002)
        assert all(abs(data_target_proportions - test_target_proportions) < 0.002)

    def test_splitters_ClassificationStratifiedDataSplitter_multiclass(self):
        holdout_ratio = 0.20
        data = TestHelper.get_iris_data()
        data = data.iloc[30:len(data)]  # data is ordered by species, so take out some from first group
        target_variable = 'species'

        data_target_proportions = dict(data.species.value_counts(normalize=True))

        test_splitter = ClassificationStratifiedDataSplitter(holdout_ratio=holdout_ratio)
        training_indexes, test_indexes = test_splitter.split(target_values=data[target_variable])

        assert isclose(len(training_indexes), len(data) * 0.8)
        assert isclose(len(test_indexes), len(data) * 0.2)
        assert set(training_indexes).isdisjoint(test_indexes)  # no overlapping indexes in training/test

        training_target_proportions = dict(data.iloc[training_indexes].species.value_counts(normalize=True))
        test_target_proportions = dict(data.iloc[test_indexes].species.value_counts(normalize=True))

        assert all([isclose(training_target_proportions[key], value) for key, value in data_target_proportions.items()])  # noqa
        assert all([isclose(test_target_proportions[key], value) for key, value in data_target_proportions.items()])  # noqa

    def test_splitters_RegressionStratifiedDataSplitter_split_monte_carlo(self):

        holdout_ratio = 0.20
        data = TestHelper.get_housing_data()
        target_variable = 'median_house_value'
        num_samples = 5

        test_splitter = RegressionStratifiedDataSplitter(holdout_ratio=holdout_ratio)
        training_indexes, test_indexes = test_splitter.split_monte_carlo(target_values=data[target_variable],
                                                                         samples=num_samples)

        assert len(training_indexes) == num_samples
        assert len(test_indexes) == num_samples

        for train_ind, test_ind, index in zip(training_indexes, test_indexes, range(num_samples)):
            assert isinstance(train_ind, list)
            assert isinstance(test_ind, list)

            # ensure sets are correct sizes and don't overlap
            assert len(train_ind) == len(data) * 0.8
            assert len(test_ind) == len(data) * 0.2
            assert set(train_ind).isdisjoint(test_ind)  # no overlapping indexes in training/test

            indexes_file_path = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_DataSplitters/RegressionStratifiedDataSplitter_monte_indexes_' + str(index) + '.pkl'))  # noqa
            # with open(indexes_file_path, 'wb') as output:
            #     pickle.dump(train_ind+test_ind, output, pickle.HIGHEST_PROTOCOL)

            with open(indexes_file_path, 'rb') as saved_object:
                training_test_index = pickle.load(saved_object)
                assert train_ind + test_ind == training_test_index

            # visualize distribution of the target variable to visually confirm stratification
            TestHelper.check_plot('data/test_DataSplitters/test_splitters_RegressionStra_monte_distribution_training_' + str(index)+'.png',  # noqa
                                  lambda: data.iloc[train_ind][target_variable].hist(color='blue', edgecolor='black', grid=None))  # noqa

            TestHelper.check_plot('data/test_DataSplitters/test_splitters_RegressionStratifi_monte_distribution_test_' + str(index) + '.png',  # noqa
                                  lambda: data.iloc[test_ind][target_variable].hist(color='blue', edgecolor='black', grid=None))  # noqa
