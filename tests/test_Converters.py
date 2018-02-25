import os
import pickle

import pandas as pd

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection SpellCheckingInspection,PyMethodMayBeStatic
class ConverterTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_TwoClassThresholdConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)

        expected_classes_50 = ['survived' if x > 0.5 else 'died' for x in predicted_probabilities['survived']]
        expected_classes_90 = ['survived' if x > 0.9 else 'died' for x in predicted_probabilities['survived']]

        converted_classes = TwoClassThresholdConverter(threshold=0.5, positive_class='survived')\
            .convert(values=predicted_probabilities)
        assert all(converted_classes == expected_classes_50)

        converted_classes_3 = TwoClassThresholdConverter(threshold=0.9, positive_class='survived')\
            .convert(values=predicted_probabilities)
        assert all(expected_classes_90 == converted_classes_3)

    def test_HighestValueConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(predicted_probabilities, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)

        # should be the same values as converting by threshold with 0.5, for the two-class case
        expected_classes = TwoClassThresholdConverter(threshold=0.5, positive_class='survived').convert(values=predicted_probabilities)  # noqa
        converted_classes = HighestValueConverter().convert(values=predicted_probabilities)
        assert all(converted_classes == expected_classes)

        # multi-class
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_MultiClass_predictions.csv')))  # noqa
        converted_classes = HighestValueConverter().convert(values=mock_data[['setosa', 'versicolor', 'virginica']])  # noqa
        assert all(converted_classes == mock_data.predicted_classes)

    def test_TwoClassRocOptimizerConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)
        actual_classes = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]  # noqa
        actual_classes = ['survived' if x == 1 else 'died' for x in actual_classes]
        # should be the same as using the TwoClassThresholdConverter directly with the ideal threshold.
        expected_classes = TwoClassThresholdConverter(threshold=0.31, positive_class='survived').convert(values=predicted_probabilities)  # noqa
        converter = TwoClassRocOptimizerConverter(actual_classes=actual_classes, positive_class='survived')
        converted_classes = converter.convert(values=predicted_probabilities)
        assert converter.ideal_threshold == 0.31
        assert all(converted_classes == expected_classes)

    def test_TwoClassPrecisionRecallOptimizerConverter(self):
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Converters/logistic_regression_output.pkl'))  # noqa
        with open(file, 'rb') as saved_object:
            predicted_probabilities = pickle.load(saved_object)
        actual_classes = [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]  # noqa
        actual_classes = ['survived' if x == 1 else 'died' for x in actual_classes]
        # should be the same as using the TwoClassThresholdConverter directly with the ideal threshold.
        expected_classes = TwoClassThresholdConverter(threshold=0.51, positive_class='survived').convert(values=predicted_probabilities)  # noqa
        converter = TwoClassPrecisionRecallOptimizerConverter(actual_classes=actual_classes, positive_class='survived')  # noqa
        converted_classes = converter.convert(values=predicted_probabilities)
        assert converter.ideal_threshold == 0.51
        assert all(converted_classes == expected_classes)
