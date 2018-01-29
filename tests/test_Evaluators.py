import os
from math import isclose
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


class MockTwoClassEvaluator(TwoClassEvaluator):
    def __init__(self,
                 positive_category,
                 negative_category,
                 use_probabilities: bool=True,
                 threshold: float=0.5):
        super().__init__(better_than=lambda this, other: this > other,  # larger value is better
                         positive_category=positive_category,
                         negative_category=negative_category,
                         use_probabilities=use_probabilities,
                         threshold=threshold)

    @property
    def metric_name(self) -> str:
        return 'Mock Evaluator'

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        return self._confusion_matrix.two_class_accuracy, self._confusion_matrix


# noinspection PyMethodMayBeStatic
class EvaluatorTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_BaseClass(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        rmse_eval = RmseEvaluator()
        accuracy = rmse_eval.evaluate(actual_values=actual, predicted_values=predicted)
        assert accuracy == 2.9154759474226504
        # should not be able to call evaluate twice on same object (could indicate some sort of reuse error)
        self.assertRaises(AssertionError,
                          lambda: rmse_eval.evaluate(actual_values=actual, predicted_values=predicted))

    def test_RmseEvaluator(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        rmse_eval = RmseEvaluator()
        assert rmse_eval.metric_name == Metric.ROOT_MEAN_SQUARE_ERROR.value
        rmse_eval.evaluate(actual_values=actual, predicted_values=predicted)
        assert isclose(2.91547594742265, rmse_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        rmse_other = RmseEvaluator()
        rmse_other.evaluate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(rmse_other.value, 3.5355339059327378)  # "worse"

        eval_list = [rmse_other, rmse_eval]  # "worse, better"
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [2.9154759474226504, 3.5355339059327378])])

    def test_MaeEvaluator(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        mae_eval = MaeEvaluator()
        assert mae_eval.metric_name == Metric.MEAN_ABSOLUTE_ERROR.value
        mae_eval.evaluate(actual_values=actual, predicted_values=predicted)
        assert isclose(2.3333333333333335, mae_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        rmse_other = RmseEvaluator()
        rmse_other.evaluate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(rmse_other.value, 3.5355339059327378)  # "worse"

        eval_list = [rmse_other, mae_eval]  # "worse, better"
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [2.3333333333333335, 3.5355339059327378])])

    def test_ConfusionMatrix_creations_result_in_same_confusion_matrix(self):
        true_positives = 21
        true_negatives = 25
        false_positives = 20
        false_negatives = 34
        negative_category = 0
        positive_category = 1

        # THESE ARE THE EXPECTED VALUES IN THE CONFUSION MATRIX (COLUMNS) FOR EACH CREATION FUNCTION
        expected_predicted_negatives = [true_negatives, false_negatives, true_negatives + false_negatives]
        expected_predicted_positives = [false_positives, true_positives, true_positives + false_positives]
        expected_totals = [sum(x) for x in zip(expected_predicted_negatives, expected_predicted_positives)]

        ######################################################################################################
        # `from_values` no categories
        ######################################################################################################
        confusion_matrix = ConfusionMatrix.from_values(true_positives=true_positives,
                                                       true_negatives=true_negatives,
                                                       false_positives=false_positives,
                                                       false_negatives=false_negatives)
        assert confusion_matrix.matrix['neg'].values.tolist() == expected_predicted_negatives
        assert confusion_matrix.matrix['pos'].values.tolist() == expected_predicted_positives
        assert confusion_matrix.matrix['Total'].values.tolist() == expected_totals

        assert confusion_matrix.matrix.index.values.tolist() == ['neg', 'pos', 'Total']
        assert confusion_matrix.matrix.columns.values.tolist() == ['neg', 'pos', 'Total']
        ######################################################################################################
        # `from_values` with categories
        ######################################################################################################
        confusion_matrix = ConfusionMatrix.from_values(true_positives=true_positives,
                                                       true_negatives=true_negatives,
                                                       false_positives=false_positives,
                                                       false_negatives=false_negatives,
                                                       positive_category=positive_category,
                                                       negative_category=negative_category)
        assert confusion_matrix.matrix.loc[:, 0].values.tolist() == expected_predicted_negatives
        assert confusion_matrix.matrix.loc[:, 1].values.tolist() == expected_predicted_positives
        assert confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == expected_totals

        assert confusion_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert confusion_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        ######################################################################################################
        # `from_values` swapped categories
        ######################################################################################################
        confusion_matrix = ConfusionMatrix.from_values(true_positives=true_negatives,
                                                       true_negatives=true_positives,
                                                       false_positives=false_negatives,
                                                       false_negatives=false_positives,
                                                       positive_category=negative_category,
                                                       negative_category=positive_category)
        expected_predicted_positives_r = [false_negatives, true_negatives, true_negatives + false_negatives]
        expected_predicted_negatives_r = [true_positives, false_positives, true_positives + false_positives]

        assert confusion_matrix.matrix.loc[:, 1].values.tolist() == expected_predicted_negatives_r
        assert confusion_matrix.matrix.loc[:, 0].values.tolist() == expected_predicted_positives_r
        assert confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [55, 45, 100]

        assert confusion_matrix.matrix.index.values.tolist() == [1, 0, 'Total']
        assert confusion_matrix.matrix.columns.values.tolist() == [1, 0, 'Total']
        ######################################################################################################
        # `from_predictions`
        ######################################################################################################
        np.random.seed(44)
        actual_values = np.random.randint(low=0, high=2, size=100)
        np.random.seed(46)
        predicted_values = np.random.randint(low=0, high=2, size=100)
        positive_category = 1

        confusion_matrix = ConfusionMatrix.from_predictions(actual_values=actual_values,
                                                            predicted_values=predicted_values,
                                                            positive_category=positive_category,
                                                            negative_category=negative_category)

        assert confusion_matrix.matrix.loc[:, 0].values.tolist() == expected_predicted_negatives
        assert confusion_matrix.matrix.loc[:, 1].values.tolist() == expected_predicted_positives
        assert confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == expected_totals

        assert confusion_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert confusion_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']

        ######################################################################################################
        # `from_predictions` swapped categories
        ######################################################################################################
        confusion_matrix = ConfusionMatrix.from_predictions(actual_values=actual_values,
                                                            predicted_values=predicted_values,
                                                            positive_category=negative_category,
                                                            negative_category=positive_category)

        assert confusion_matrix.matrix.loc[:, 1].values.tolist() == expected_predicted_negatives_r
        assert confusion_matrix.matrix.loc[:, 0].values.tolist() == expected_predicted_positives_r
        assert confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [55, 45, 100]

        assert confusion_matrix.matrix.index.values.tolist() == [1, 0, 'Total']
        assert confusion_matrix.matrix.columns.values.tolist() == [1, 0, 'Total']

    def check_confusion_matrix(self, con_matrix, mock_data):
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [347, 140, 487]
        assert con_matrix.matrix.loc[:, 1].values.tolist() == [77, 150, 227]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        assert isclose(con_matrix.all_quality_metrics['Kappa'], 0.34756903797404387)
        assert isclose(con_matrix.all_quality_metrics['Two-Class Accuracy'], 0.69607843137254899)
        assert isclose(con_matrix.all_quality_metrics['Error Rate'], 0.30392156862745096)
        assert isclose(con_matrix.all_quality_metrics['Sensitivity'], 0.51724137931034486)
        assert isclose(con_matrix.all_quality_metrics['Specificity'], 0.81839622641509435)
        assert isclose(con_matrix.all_quality_metrics['False Positive Rate'], 1 - con_matrix.specificity)
        assert isclose(con_matrix.all_quality_metrics['False Negative Rate'], 1 - con_matrix.sensitivity)
        assert isclose(con_matrix.all_quality_metrics['Positive Predictive Value'], 0.66079295154185025)
        assert isclose(con_matrix.all_quality_metrics['Negative Predictive Value'], 0.71252566735112932)
        assert isclose(con_matrix.all_quality_metrics['Prevalence'], 0.4061624649859944)
        assert isclose(con_matrix.all_quality_metrics['No Information Rate'], 0.5938375350140056)
        assert isclose(con_matrix.all_quality_metrics['Total Observations'], len(mock_data))

    def test_ConfusionMatrix_correct_calculations(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        ######################################################################################################
        # `from_predictions` check calculations (verified against R's caret.confusionMatrix
        ######################################################################################################
        con_matrix = ConfusionMatrix.from_predictions(actual_values=mock_data.actual,
                                                      predicted_values=mock_data.predictions,
                                                      positive_category=1,
                                                      negative_category=0)

        self.check_confusion_matrix(con_matrix, mock_data)

        ######################################################################################################
        # `from_predictions` check calculations SWAPPED
        ######################################################################################################
        con_matrix = ConfusionMatrix.from_predictions(actual_values=mock_data.actual,
                                                      predicted_values=mock_data.predictions,
                                                      positive_category=0,
                                                      negative_category=1)

        assert con_matrix.matrix.loc[:, 1].values.tolist() == [150, 77, 227]
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [140, 347, 487]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [290, 424, 714]
        assert con_matrix.matrix.index.values.tolist() == [1, 0, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [1, 0, 'Total']

        assert isclose(con_matrix.all_quality_metrics['Kappa'], 0.34756903797404387)
        assert isclose(con_matrix.all_quality_metrics['Two-Class Accuracy'], 0.69607843137254899)
        assert isclose(con_matrix.all_quality_metrics['Error Rate'], 0.30392156862745096)
        assert isclose(con_matrix.all_quality_metrics['Sensitivity'], 0.81839622641509435)
        assert isclose(con_matrix.all_quality_metrics['Specificity'], 0.51724137931034486)
        assert isclose(con_matrix.all_quality_metrics['False Positive Rate'], 1 - con_matrix.specificity)
        assert isclose(con_matrix.all_quality_metrics['False Negative Rate'], 1 - con_matrix.sensitivity)
        assert isclose(con_matrix.all_quality_metrics['Positive Predictive Value'], 0.71252566735112932)
        assert isclose(con_matrix.all_quality_metrics['Negative Predictive Value'], 0.66079295154185025)
        assert isclose(con_matrix.all_quality_metrics['Prevalence'], 1 - 0.4061624649859944)
        assert isclose(con_matrix.all_quality_metrics['No Information Rate'], 1 - 0.5938375350140056)
        assert isclose(con_matrix.all_quality_metrics['Total Observations'], len(mock_data))

    def test_TwoClassEvaluator_predictions(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        # since target 'category' is 0/1, round() the probabilities will select the right category
        assert all(mock_data.predictions == round(mock_data.pos_probabilities))  # ensure correct data

        evaluator = MockTwoClassEvaluator(positive_category=1, negative_category=0, use_probabilities=False)
        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=mock_data.predictions)
        assert isclose(accuracy, 0.69607843137254899)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix)
        self.check_confusion_matrix(con_matrix=evaluator.confusion_matrix, mock_data=mock_data)
        assert evaluator.value == evaluator.confusion_matrix.two_class_accuracy
        assert evaluator.threshold is None

    def test_TwoClassEvaluator_probabilities_custom_threshold(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(['actual', 'predictions'], axis=1)
        predictions_mock.columns = [1, 0]

        evaluator = MockTwoClassEvaluator(positive_category=1, negative_category=0, use_probabilities=True,
                                          threshold=0.5)
        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix)
        self.check_confusion_matrix(con_matrix=evaluator.confusion_matrix, mock_data=mock_data)
        assert evaluator.value == evaluator.confusion_matrix.two_class_accuracy
        assert evaluator.threshold == 0.5
        assert isclose(evaluator.auc, 0.74428675992192583)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_TwoClassEvaluator_probabilities_custom_thr_ROC.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        evaluator.get_roc_curve()
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)

    def test_TwoClassEvaluator_probabilities_no_threshold(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(['actual', 'predictions'], axis=1)
        predictions_mock.columns = [1, 0]

        # noinspection PyTypeChecker
        evaluator = MockTwoClassEvaluator(positive_category=1,
                                          negative_category=0,
                                          use_probabilities=True,
                                          threshold=None)
        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix)
        assert evaluator.confusion_matrix.matrix.loc[:, 0].values.tolist() == [296, 90, 386]
        assert evaluator.confusion_matrix.matrix.loc[:, 1].values.tolist() == [128, 200, 328]
        assert evaluator.confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator.value == evaluator.confusion_matrix.two_class_accuracy
        assert isclose(evaluator.threshold, 0.41)
        assert isclose(evaluator.auc, 0.74428675992192583)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_TwoClassEvaluator_probabilities_no_thresh_ROC.png'))  # noqa
        assert os.path.isfile(file)
        os.remove(file)
        assert os.path.isfile(file) is False
        evaluator.get_roc_curve()
        plt.savefig(file)
        plt.gcf().clear()
        assert os.path.isfile(file)


    def test_AucEvaluator(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(['actual', 'predictions'], axis=1)
        predictions_mock.columns = [1, 0]

        # noinspection PyTypeChecker
        evaluator = AucEvaluator(positive_category=1,
                                 negative_category=0,
                                 use_probabilities=True,
                                 threshold=None)
        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix)
        assert evaluator.confusion_matrix.matrix.loc[:, 0].values.tolist() == [296, 90, 386]
        assert evaluator.confusion_matrix.matrix.loc[:, 1].values.tolist() == [128, 200, 328]
        assert evaluator.confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator.value == evaluator.auc
        assert isclose(evaluator.threshold, 0.41)
        assert isclose(evaluator.auc, 0.74428675992192583)
        ######################################################################################################
        # Test sorting
        ######################################################################################################
        evaluator_other = AucEvaluator(positive_category=0,
                                       negative_category=1,
                                       use_probabilities=True,
                                       threshold=0.5)  # creates worse value
        accuracy = evaluator_other.evaluate(actual_values=mock_data.actual,
                                            predicted_values=predictions_mock)
        assert isclose(accuracy, 0.25571324007807417)  # lower number means it is worse than first value

        eval_list = [evaluator_other, evaluator]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.25571324007807417, 0.74428675992192583])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.74428675992192583, 0.25571324007807417])])

    # noinspection PyTypeChecker
    def test_KappaEvaluator(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(['actual', 'predictions'], axis=1)
        predictions_mock.columns = [1, 0]

        evaluator = KappaEvaluator(positive_category=1,
                                   negative_category=0,
                                   use_probabilities=True,
                                   threshold=None)
        accuracy = evaluator.evaluate(actual_values=mock_data.actual,
                                      predicted_values=predictions_mock)
        assert isclose(accuracy, 0.37990215607221967)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix)
        assert evaluator.confusion_matrix.matrix.loc[:, 0].values.tolist() == [296, 90, 386]
        assert evaluator.confusion_matrix.matrix.loc[:, 1].values.tolist() == [128, 200, 328]
        assert evaluator.confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator.value == evaluator.confusion_matrix.kappa
        assert isclose(evaluator.threshold, 0.41)
        assert isclose(evaluator.auc, 0.74428675992192583)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        evaluator_other = KappaEvaluator(positive_category=0,
                                         negative_category=1,
                                         use_probabilities=True,
                                         threshold=0.5)  # creates worse value
        accuracy = evaluator_other.evaluate(actual_values=mock_data.actual,
                                            predicted_values=predictions_mock)
        assert isclose(accuracy, 0.34756903797404387)  # lower number means it is worse than first value

        eval_list = [evaluator_other, evaluator]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.34756903797404387, 0.37990215607221967])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.37990215607221967, 0.34756903797404387])])

    def test_Misc_evaluators(self):
        """
        For example, these holdout_evaluators might be already tested in another class (e.g. Sensitivity is
            tested via ConfusionMatrix), but we want to verify we can instantiate and use.
        """
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(['actual', 'predictions'], axis=1)
        predictions_mock.columns = [1, 0]

        evaluator = SensitivityEvaluator(positive_category=0,
                                         negative_category=1,
                                         use_probabilities=True,
                                         threshold=0.5)  # creates worse value
        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.81839622641509435)  # lower number means it is worse than first value
        evaluator = SpecificityEvaluator(positive_category=0,
                                         negative_category=1,
                                         use_probabilities=True,
                                         threshold=0.5)  # creates worse value
        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.51724137931034486)  # lower number means it is worse than first value
