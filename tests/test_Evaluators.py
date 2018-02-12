import os
from math import isclose
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, \
    mean_squared_error, mean_absolute_error, roc_auc_score
from typing import Tuple

import dill as pickle
import numpy as np
import pandas as pd

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


class MockTwoClassEvaluator(UtilityFunctionMixin, TwoClassEvaluator):
    def __init__(self,
                 positive_category,
                 negative_category,
                 use_probabilities: bool=True,
                 threshold: float=0.5):
        super().__init__(positive_category=positive_category,
                         negative_category=negative_category,
                         use_probabilities=use_probabilities,
                         threshold=threshold)

    @property
    def metric_name(self) -> str:
        return 'Mock Evaluator'

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        return self._confusion_matrix.two_class_accuracy, self._confusion_matrix


class MockMultiClassEvaluator(UtilityFunctionMixin, MultiClassEvaluator):
    @property
    def metric_name(self) -> str:
        return 'Mock Evaluator'

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        return self._confusion_matrix.accuracy, self._confusion_matrix


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

        assert isinstance(rmse_eval, CostFunctionMixin)

    def test_RmseEvaluator(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        rmse_eval = RmseEvaluator()
        assert isinstance(rmse_eval, CostFunctionMixin)
        assert isinstance(rmse_eval, EvaluatorBase)
        assert rmse_eval.metric_name == Metric.ROOT_MEAN_SQUARE_ERROR.value
        rmse_eval.evaluate(actual_values=actual, predicted_values=predicted)
        assert isclose(np.sqrt(mean_squared_error(y_true=actual, y_pred=predicted)), rmse_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        rmse_other = RmseEvaluator()
        rmse_other.evaluate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(rmse_other.value, 3.5355339059327378)  # "worse"
        eval_list = [rmse_other, rmse_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [3.5355339059327378, 2.9154759474226504])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [2.9154759474226504, 3.5355339059327378])])

    def test_MaeEvaluator(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        mae_eval = MaeEvaluator()
        assert isinstance(mae_eval, CostFunctionMixin)
        assert isinstance(mae_eval, EvaluatorBase)
        assert mae_eval.metric_name == Metric.MEAN_ABSOLUTE_ERROR.value
        mae_eval.evaluate(actual_values=actual, predicted_values=predicted)
        assert isclose(mean_absolute_error(y_true=actual, y_pred=predicted), mae_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        rmse_other = RmseEvaluator()
        rmse_other.evaluate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(rmse_other.value, 3.5355339059327378)  # "worse"

        eval_list = [rmse_other, mae_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [3.5355339059327378, 2.3333333333333335])])
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
        confusion_matrix = ConfusionMatrix2C.from_values(true_positives=true_positives,
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
        confusion_matrix = ConfusionMatrix2C.from_values(true_positives=true_positives,
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
        confusion_matrix = ConfusionMatrix2C.from_values(true_positives=true_negatives,
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
        # `from_classes`
        ######################################################################################################
        np.random.seed(44)
        actual_values = np.random.randint(low=0, high=2, size=100)
        np.random.seed(46)
        predicted_values = np.random.randint(low=0, high=2, size=100)
        positive_category = 1

        confusion_matrix = ConfusionMatrix2C.from_classes(actual_classes=actual_values,
                                                          predicted_classes=predicted_values,
                                                          positive_category=positive_category,
                                                          negative_category=negative_category)

        assert confusion_matrix.matrix.loc[:, 0].values.tolist() == expected_predicted_negatives
        assert confusion_matrix.matrix.loc[:, 1].values.tolist() == expected_predicted_positives
        assert confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == expected_totals

        assert confusion_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert confusion_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']

        ######################################################################################################
        # `from_classes` swapped categories
        ######################################################################################################
        confusion_matrix = ConfusionMatrix2C.from_classes(actual_classes=actual_values,
                                                          predicted_classes=predicted_values,
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
        assert isclose(con_matrix.all_quality_metrics['Kappa'], cohen_kappa_score(y1=mock_data.actual, y2=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['F1 Score'], f1_score(y_true=mock_data.actual, y_pred=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Two-Class Accuracy'], accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Error Rate'], 1 - accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['True Positive Rate'], recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['True Negative Rate'], recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions, pos_label=0))  # noqa
        assert isclose(con_matrix.all_quality_metrics['False Positive Rate'], 1 - con_matrix.specificity)
        assert isclose(con_matrix.all_quality_metrics['False Negative Rate'], 1 - con_matrix.sensitivity)
        assert isclose(con_matrix.all_quality_metrics['Positive Predictive Value'], precision_score(y_true=mock_data.actual, y_pred=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Negative Predictive Value'], precision_score(y_true=mock_data.actual, y_pred=mock_data.predictions, pos_label=0))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Prevalence'], 0.4061624649859944)
        assert isclose(con_matrix.all_quality_metrics['No Information Rate'], 0.5938375350140056)
        assert isclose(con_matrix.all_quality_metrics['Total Observations'], len(mock_data))

    def test_ConfusionMatrix_correct_calculations(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        ######################################################################################################
        # `from_classes` check calculations (verified against R's caret.confusionMatrix
        ######################################################################################################
        con_matrix = ConfusionMatrix2C.from_classes(actual_classes=mock_data.actual,
                                                    predicted_classes=mock_data.predictions,
                                                    positive_category=1,
                                                    negative_category=0)

        self.check_confusion_matrix(con_matrix, mock_data)

        ######################################################################################################
        # `from_classes` check calculations *******SWAPPED********
        ######################################################################################################
        con_matrix = ConfusionMatrix2C.from_classes(actual_classes=mock_data.actual,
                                                    predicted_classes=mock_data.predictions,
                                                    positive_category=0,
                                                    negative_category=1)

        assert con_matrix.matrix.loc[:, 1].values.tolist() == [150, 77, 227]
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [140, 347, 487]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [290, 424, 714]
        assert con_matrix.matrix.index.values.tolist() == [1, 0, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [1, 0, 'Total']

        assert isclose(con_matrix.all_quality_metrics['Kappa'], cohen_kappa_score(y1=mock_data.actual, y2=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['F1 Score'], f1_score(y_true=mock_data.actual, y_pred=mock_data.predictions, pos_label=0))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Two-Class Accuracy'], accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Error Rate'], 1 - accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))  # noqa
        assert isclose(con_matrix.all_quality_metrics['True Positive Rate'], recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions, pos_label=0))  # noqa
        assert isclose(con_matrix.all_quality_metrics['True Negative Rate'], recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions, pos_label=1))  # noqa
        assert isclose(con_matrix.all_quality_metrics['False Positive Rate'], 1 - con_matrix.specificity)
        assert isclose(con_matrix.all_quality_metrics['False Negative Rate'], 1 - con_matrix.sensitivity)
        assert isclose(con_matrix.all_quality_metrics['Positive Predictive Value'], precision_score(y_true=mock_data.actual, y_pred=mock_data.predictions, pos_label=0))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Negative Predictive Value'], precision_score(y_true=mock_data.actual, y_pred=mock_data.predictions, pos_label=1))  # noqa
        assert isclose(con_matrix.all_quality_metrics['Prevalence'], 1 - 0.4061624649859944)
        assert isclose(con_matrix.all_quality_metrics['No Information Rate'], 0.5938375350140056)
        assert isclose(con_matrix.all_quality_metrics['Total Observations'], len(mock_data))

    def test_ConfusionMatrix_from_probabilities(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        con_matrix = ConfusionMatrix2C.from_probabilities(actual_classes=mock_data.actual,
                                                          predicted_probabilities=mock_data[['pos_probabilities', 'neg_probabilities']].rename(columns={'pos_probabilities': 1, 'neg_probabilities': 0}),  # noqa
                                                          positive_category=1,
                                                          negative_category=0)
        self.check_confusion_matrix(con_matrix, mock_data)
        con_matrix = ConfusionMatrix2C.from_probabilities(actual_classes=mock_data.actual,
                                                          predicted_probabilities=mock_data[['pos_probabilities', 'neg_probabilities']].rename(columns={'pos_probabilities': 1, 'neg_probabilities': 0}),  # noqa
                                                          positive_category=1,
                                                          negative_category=0,
                                                          threshold=0.5)
        self.check_confusion_matrix(con_matrix, mock_data)

        ######################################################################################################
        # try a threshold of 1, which means that 0 positives will be predicted
        ######################################################################################################
        con_matrix = ConfusionMatrix2C.from_probabilities(actual_classes=mock_data.actual,
                                                          predicted_probabilities=mock_data[['pos_probabilities', 'neg_probabilities']].rename(columns={'pos_probabilities': 1, 'neg_probabilities': 0}),  # noqa
                                                          positive_category=1,
                                                          negative_category=0,
                                                          threshold=1)
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [424, 290, 714]
        assert con_matrix.matrix.loc[:, 1].values.tolist() == [0, 0, 0]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        assert isclose(con_matrix.all_quality_metrics['Kappa'], 0)
        assert con_matrix.all_quality_metrics['F1 Score'] is None
        assert isclose(con_matrix.all_quality_metrics['Two-Class Accuracy'], con_matrix.negative_predictive_value)  # noqa
        assert isclose(con_matrix.all_quality_metrics['Error Rate'], con_matrix.prevalence)  # noqa
        assert isclose(con_matrix.all_quality_metrics['True Positive Rate'], 0)
        assert isclose(con_matrix.all_quality_metrics['True Negative Rate'], 1)
        assert isclose(con_matrix.all_quality_metrics['False Positive Rate'], 0)
        assert isclose(con_matrix.all_quality_metrics['False Negative Rate'], 1)
        assert con_matrix.all_quality_metrics['Positive Predictive Value'] is None
        assert isclose(con_matrix.all_quality_metrics['Negative Predictive Value'], con_matrix.all_quality_metrics['No Information Rate'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['Prevalence'], 0.4061624649859944)
        assert isclose(con_matrix.all_quality_metrics['No Information Rate'], 0.5938375350140056)
        assert isclose(con_matrix.all_quality_metrics['Total Observations'], len(mock_data))

    def test_TwoClassEvaluator_predictions(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        # since target 'category' is 0/1, round() the probabilities will select the right category
        assert all(mock_data.predictions == round(mock_data.pos_probabilities))  # ensure correct data

        evaluator = MockTwoClassEvaluator(positive_category=1, negative_category=0, use_probabilities=False)
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)

        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=mock_data.predictions)
        assert isclose(accuracy, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix2C)
        self.check_confusion_matrix(con_matrix=evaluator.confusion_matrix, mock_data=mock_data)
        assert evaluator.value == evaluator.confusion_matrix.two_class_accuracy
        assert evaluator.threshold is None

    def test_TwoClassEvaluator_probabilities_custom_threshold(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        evaluator = MockTwoClassEvaluator(positive_category=1, negative_category=0, use_probabilities=True,
                                          threshold=0.5)
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)

        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix2C)
        self.check_confusion_matrix(con_matrix=evaluator.confusion_matrix, mock_data=mock_data)
        assert evaluator.value == evaluator.confusion_matrix.two_class_accuracy
        assert evaluator.threshold == 0.5
        assert isclose(evaluator.auc, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa

        actual_thresholds = evaluator._calculate_fpr_tpr_ideal_threshold()
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/fpr_tpr_threshold_mock.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(actual_thresholds, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_thresholds = pickle.load(saved_object)
            assert len(expected_thresholds) == 3
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[0], actual_thresholds[0])])
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[1], actual_thresholds[1])])
            assert isclose(expected_thresholds[2], actual_thresholds[2])

        actual_thresholds = evaluator._calculate_ppv_tpr_ideal_threshold()
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/ppv_tpr_threshold_mock.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(actual_thresholds, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_thresholds = pickle.load(saved_object)
            assert len(expected_thresholds) == 3
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[0], actual_thresholds[0])])
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[1], actual_thresholds[1])])
            assert isclose(expected_thresholds[2], actual_thresholds[2])

        # import time
        # started_at = time.time()
        # evaluator._calculate_fpr_tpr_ideal_threshold()
        # elapsed = time.time() - started_at
        # assert elapsed < 0.5

        TestHelper.check_plot('data/test_Evaluators/test_TwoClassEvaluator_probabilities_custom_thr_ROC.png',  # noqa
                              lambda: evaluator.get_roc_curve())

        TestHelper.check_plot('data/test_Evaluators/test_TwoClassEvaluator_probabilities_custom_thr_ppv_tpr.png',  # noqa
                              lambda: evaluator.get_ppv_tpr_curve())

    def test_TwoClassEvaluator_probabilities_no_threshold(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        # noinspection PyTypeChecker
        evaluator = MockTwoClassEvaluator(positive_category=1,
                                          negative_category=0,
                                          use_probabilities=True,
                                          threshold=None)
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)

        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix2C)
        assert evaluator.confusion_matrix.matrix.loc[:, 0].values.tolist() == [296, 90, 386]
        assert evaluator.confusion_matrix.matrix.loc[:, 1].values.tolist() == [128, 200, 328]
        assert evaluator.confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator.value == evaluator.confusion_matrix.two_class_accuracy
        assert isclose(evaluator.threshold, 0.41)
        assert isclose(evaluator.auc, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa

        actual_thresholds = evaluator._calculate_fpr_tpr_ideal_threshold()
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/fpr_tpr_threshold_mock.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(actual_thresholds, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_thresholds = pickle.load(saved_object)
            assert len(expected_thresholds) == 3
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[0], actual_thresholds[0])])
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[1], actual_thresholds[1])])
            assert isclose(expected_thresholds[2], actual_thresholds[2])

        actual_thresholds = evaluator._calculate_ppv_tpr_ideal_threshold()
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/ppv_tpr_threshold_mock.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(actual_thresholds, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_thresholds = pickle.load(saved_object)
            assert len(expected_thresholds) == 3
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[0], actual_thresholds[0])])
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[1], actual_thresholds[1])])
            assert isclose(expected_thresholds[2], actual_thresholds[2])

        TestHelper.check_plot('data/test_Evaluators/test_TwoClassEvaluator_probabilities_no_thresh_ROC.png',  # noqa
                              lambda: evaluator.get_roc_curve())

        TestHelper.check_plot('data/test_Evaluators/test_TwoClassEvaluator_probabilities_no_thr_ppv_tpr.png',  # noqa
                              lambda: evaluator.get_ppv_tpr_curve())

    def test_AucEvaluator(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        # noinspection PyTypeChecker
        evaluator = AucEvaluator(positive_category=1,
                                 negative_category=0,
                                 use_probabilities=True,
                                 threshold=None)
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)

        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix2C)
        assert evaluator.confusion_matrix.matrix.loc[:, 0].values.tolist() == [296, 90, 386]
        assert evaluator.confusion_matrix.matrix.loc[:, 1].values.tolist() == [128, 200, 328]
        assert evaluator.confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator.value == evaluator.auc
        assert isclose(evaluator.threshold, 0.41)
        assert isclose(evaluator.auc, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa
        ######################################################################################################
        # Test sorting
        ######################################################################################################
        evaluator_other = AucEvaluator(positive_category=0,
                                       negative_category=1,
                                       use_probabilities=True,
                                       threshold=0.5)  # creates worse value
        accuracy = evaluator_other.evaluate(actual_values=mock_data.actual,
                                            predicted_values=predictions_mock)
        assert isclose(accuracy, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.neg_probabilities))

        eval_list = [evaluator_other, evaluator]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.25571324007807417, 0.74428675992192583])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.74428675992192583, 0.25571324007807417])])

    def test_KappaEvaluator(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]
        ######################################################################################################
        # NOTE: because we are setting `threshold=None`, it means the threshold will be calculated, which
        # will change the values of the kappa/f1/etc. from the default threshold of 0.5; but, e.g. the AUC
        # will not change
        ######################################################################################################
        evaluator = KappaEvaluator(positive_category=1,
                                   negative_category=0,
                                   use_probabilities=True,
                                   threshold=None)
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)

        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.37990215607221967)  # will be different Kappa than sklearn, from threshold
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix2C)
        assert evaluator.confusion_matrix.matrix.loc[:, 0].values.tolist() == [296, 90, 386]
        assert evaluator.confusion_matrix.matrix.loc[:, 1].values.tolist() == [128, 200, 328]
        assert evaluator.confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator.value == evaluator.confusion_matrix.kappa
        assert isclose(evaluator.threshold, 0.41)
        assert isclose(evaluator.auc, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        evaluator_other = KappaEvaluator(positive_category=0,
                                         negative_category=1,
                                         use_probabilities=True,
                                         threshold=0.5)  # creates worse value
        accuracy = evaluator_other.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, cohen_kappa_score(y1=mock_data.actual, y2=mock_data.predictions))

        eval_list = [evaluator_other, evaluator]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.34756903797404387, 0.37990215607221967])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.37990215607221967, 0.34756903797404387])])

    def test_F1Evaluator(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        ######################################################################################################
        # NOTE: because we are setting `threshold=None`, it means the threshold will be calculated, which
        # will change the values of the kappa/f1/etc. from the default threshold of 0.5; but, e.g. the AUC
        # will not change
        ######################################################################################################
        evaluator = F1Evaluator(positive_category=1,
                                negative_category=0,
                                use_probabilities=True,
                                threshold=None)
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)

        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.6472491909385113)  # will be different Kappa than sklearn, from threshold
        assert isinstance(evaluator.confusion_matrix, ConfusionMatrix2C)
        assert evaluator.confusion_matrix.matrix.loc[:, 0].values.tolist() == [296, 90, 386]
        assert evaluator.confusion_matrix.matrix.loc[:, 1].values.tolist() == [128, 200, 328]
        assert evaluator.confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator.value == evaluator.confusion_matrix.f1_score
        assert isclose(evaluator.threshold, 0.41)
        assert isclose(evaluator.auc, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        evaluator_other = F1Evaluator(positive_category=0,
                                      negative_category=1,
                                      use_probabilities=True,
                                      threshold=0.5)  # creates worse value
        accuracy = evaluator_other.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, f1_score(y_true=mock_data.actual, y_pred=mock_data.predictions,  pos_label=0))  # noqa

        eval_list = [evaluator, evaluator_other]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.6472491909385113, 0.7618002195389681])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.7618002195389681, 0.6472491909385113])])

    def test_Misc_evaluators(self):
        """
        For example, these holdout_evaluators might be already tested in another class (e.g. Sensitivity is
            tested via ConfusionMatrix2C), but we want to verify we can instantiate and use.
        """
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]
        ######################################################################################################
        evaluator = SensitivityEvaluator(positive_category=1,
                                         negative_category=0,
                                         use_probabilities=True,
                                         threshold=0.5)  # creates worse value
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)
        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions))
        assert isclose(evaluator.value, recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions))
        ######################################################################################################
        evaluator = SpecificityEvaluator(positive_category=1,
                                         negative_category=0,
                                         use_probabilities=True,
                                         threshold=0.5)  # creates worse value
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)
        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 1 - evaluator.confusion_matrix.false_positive_rate)
        assert isclose(evaluator.value, 1 - evaluator.confusion_matrix.false_positive_rate)
        ######################################################################################################
        evaluator = Accuracy2CEvaluator(positive_category=1,
                                        negative_category=0,
                                        use_probabilities=True,
                                        threshold=0.5)
        assert isinstance(evaluator, UtilityFunctionMixin)
        assert isinstance(evaluator, EvaluatorBase)
        accuracy = evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))
        assert isclose(evaluator.value, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))

    # noinspection SpellCheckingInspection
    def test_ConfusionMatrix_MultiClass(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_MultiClass_predictions.csv')))  # noqa
        con_matrix = ConfusionMatrix(actual_classes=mock_data.actual, predicted_classes=mock_data.predicted_classes)  # noqa

        assert con_matrix.matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix.matrix['versicolor'].values.tolist() == [0, 12, 2, 14]
        assert con_matrix.matrix['virginica'].values.tolist() == [0, 1, 11, 12]
        assert con_matrix.matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

        ######################################################################################################
        # change all setosa predictions to versicolor
        # i.e. there will be no predictions for a class (setosa), make sure Confusion Matrix can handle that.
        ######################################################################################################
        no_setosa = np.array([x if x != 'setosa' else 'versicolor' for x in mock_data.predicted_classes])
        con_matrix = ConfusionMatrix(actual_classes=mock_data.actual, predicted_classes=no_setosa)  # noqa

        assert con_matrix.matrix['setosa'].values.tolist() == [0, 0, 0, 0]
        assert con_matrix.matrix['versicolor'].values.tolist() == [12, 12, 2, 26]
        assert con_matrix.matrix['virginica'].values.tolist() == [0, 1, 11, 12]
        assert con_matrix.matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

        ######################################################################################################
        # test from probabilities
        ######################################################################################################
        con_matrix = ConfusionMatrix.from_probabilities(actual_classes=mock_data.actual,
                                                        predicted_probabilities=mock_data[['setosa', 'versicolor', 'virginica']])  # noqa
        assert con_matrix.matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert con_matrix.matrix['versicolor'].values.tolist() == [0, 12, 2, 14]
        assert con_matrix.matrix['virginica'].values.tolist() == [0, 1, 11, 12]
        assert con_matrix.matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert con_matrix.matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert con_matrix.matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

    def test_ConfusionMatrix_MultiClass_scores(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_MultiClass_predictions.csv')))  # noqa
        con_matrix = ConfusionMatrix(actual_classes=mock_data.actual, predicted_classes=mock_data.predicted_classes)  # noqa
        assert isclose(con_matrix.accuracy, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predicted_classes))  # noqa

        assert isclose(con_matrix.all_quality_metrics['Kappa'], 0.8814968814968815)
        assert isclose(con_matrix.all_quality_metrics['Accuracy'], 0.9210526315789473)
        assert isclose(con_matrix.all_quality_metrics['Error Rate'], 0.07894736842105265)
        assert isclose(con_matrix.all_quality_metrics['No Information Rate'], 0.34210526315789475)
        assert isclose(con_matrix.all_quality_metrics['Total Observations'], 38)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/mock_metrics_per_class.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(con_matrix.metrics_per_class, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_metrics_per_class = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_metrics_per_class,
                                                      data_frame2=con_matrix.metrics_per_class)
