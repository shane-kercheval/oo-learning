import os
import pickle

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection SpellCheckingInspection,PyMethodMayBeStatic
class DataSplittersTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_TwoClassThresholdConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)

        expected_classes_50 = ['survived' if x > 0.5 else 'died' for x in predicted_probabilities['survived']]
        expected_classes_90 = ['survived' if x > 0.9 else 'died' for x in predicted_probabilities['survived']]

        converted_classes = TwoClassThresholdConverter(threshold=0.5)\
            .convert(predicted_probabilities=predicted_probabilities, positive_class='survived')
        assert converted_classes == expected_classes_50

        converted_classes_3 = TwoClassThresholdConverter(threshold=0.9)\
            .convert(predicted_probabilities=predicted_probabilities, positive_class='survived')
        assert expected_classes_90 == converted_classes_3

    def test_HighestValueConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(predicted_probabilities, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)

        # should be the same values as converting by threshold with 0.5, for the two-class case
        expected_classes = TwoClassThresholdConverter(threshold=0.5).convert(predicted_probabilities=predicted_probabilities, positive_class='survived')  # noqa
        # shouldn't pass in any other parameters
        self.assertRaises(ValueError,
                          lambda: HighestValueConverter().convert(values=predicted_probabilities,
                                                                  positive_class='survived'))

        converted_classes = HighestValueConverter().convert(values=predicted_probabilities)
        assert converted_classes == expected_classes

        # multi-class
        expected_classes = ['setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'setosa', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'virginica', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica', 'versicolor', 'virginica', 'virginica', 'virginica', 'virginica']  # noqa
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/random_forest_multiclass_output.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)

        converted_classes = HighestValueConverter().convert(values=predicted_probabilities)
        assert converted_classes == expected_classes

    def test_TwoClassRocOptimizerConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)
        actual_classes = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]  # noqa
        actual_classes = ['survived' if x == 1 else 'died' for x in actual_classes]
        # should be the same as using the TwoClassThresholdConverter directly with the ideal threshold.
        expected_classes = TwoClassThresholdConverter(threshold=0.31).convert(predicted_probabilities=predicted_probabilities, positive_class='survived')  # noqa
        converter = TwoClassRocOptimizerConverter(actual_classes=actual_classes)
        converted_classes = converter.convert(predicted_probabilities=predicted_probabilities,
                                              positive_class='survived')
        assert converter.ideal_threshold == 0.31
        assert converted_classes == expected_classes

    def test_TwoClassPrecisionRecallOptimizerConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)
        actual_classes = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]  # noqa
        actual_classes = ['survived' if x == 1 else 'died' for x in actual_classes]
        # should be the same as using the TwoClassThresholdConverter directly with the ideal threshold.
        expected_classes = TwoClassThresholdConverter(threshold=0.51).convert(predicted_probabilities=predicted_probabilities, positive_class='survived')  # noqa
        converter = TwoClassPrecisionRecallOptimizerConverter(actual_classes=actual_classes)
        converted_classes = converter.convert(predicted_probabilities=predicted_probabilities,
                                              positive_class='survived')
        assert converter.ideal_threshold == 0.51
        assert converted_classes == expected_classes
