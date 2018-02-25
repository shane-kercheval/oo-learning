import os
from math import isclose

import dill as pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, cohen_kappa_score, \
    roc_auc_score

from oolearning import *
from oolearning.evaluators.TwoClassConfusionMatrix import TwoClassConfusionMatrix
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic
class EvaluatorTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_TwoClassEvaluator_ConfusionMatrix(self):
        true_positives = 21
        true_negatives = 25
        false_positives = 20
        false_negatives = 34
        negative_category = 0

        # THESE ARE THE EXPECTED VALUES IN THE CONFUSION MATRIX (COLUMNS) FOR EACH CREATION FUNCTION
        expected_predicted_negatives = [true_negatives, false_negatives, true_negatives + false_negatives]
        expected_predicted_positives = [false_positives, true_positives, true_positives + false_positives]
        expected_totals = [sum(x) for x in zip(expected_predicted_negatives, expected_predicted_positives)]

        expected_predicted_positives_r = [false_negatives, true_negatives, true_negatives + false_negatives]
        expected_predicted_negatives_r = [true_positives, false_positives, true_positives + false_positives]

        ######################################################################################################
        # `from_classes`
        ######################################################################################################
        np.random.seed(44)
        actual_values = np.random.randint(low=0, high=2, size=100)
        np.random.seed(46)
        predicted_values = np.random.randint(low=0, high=2, size=100)
        positive_category = 1

        evaluator = TwoClassEvaluator.from_classes(actual_classes=actual_values,
                                                   predicted_classes=predicted_values,
                                                   positive_class=positive_category)

        assert evaluator.matrix.loc[:, 0].values.tolist() == expected_predicted_negatives
        assert evaluator.matrix.loc[:, 1].values.tolist() == expected_predicted_positives
        assert evaluator.matrix.loc[:, 'Total'].values.tolist() == expected_totals
        assert evaluator.total_observations == 100
        assert evaluator.confusion_matrix.total_observations == 100

        assert evaluator.matrix.index.values.tolist() == [0, 1, 'Total']
        assert evaluator.matrix.columns.values.tolist() == [0, 1, 'Total']

        ######################################################################################################
        # `from_classes` swapped categories
        ######################################################################################################
        evaluator = TwoClassEvaluator.from_classes(actual_classes=actual_values,
                                                   predicted_classes=predicted_values,
                                                   positive_class=negative_category)

        assert evaluator.matrix.loc[:, 1].values.tolist() == expected_predicted_negatives_r
        assert evaluator.matrix.loc[:, 0].values.tolist() == expected_predicted_positives_r
        assert evaluator.matrix.loc[:, 'Total'].values.tolist() == [55, 45, 100]

        assert evaluator.matrix.index.values.tolist() == [1, 0, 'Total']
        assert evaluator.matrix.columns.values.tolist() == [1, 0, 'Total']

    def check_confusion_matrix(self, con_matrix, mock_data):
        assert con_matrix.total_observations == 714
        assert con_matrix.matrix.loc[:, 0].values.tolist() == [347, 140, 487]
        assert con_matrix.matrix.loc[:, 1].values.tolist() == [77, 150, 227]
        assert con_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert con_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']

        assert con_matrix.matrix_proportions[0].values.tolist() == [0.48599439775910364, 0.19607843137254902, 0.6820728291316527]  # noqa
        assert con_matrix.matrix_proportions[1].values.tolist() == [0.10784313725490197, 0.21008403361344538, 0.3179271708683473]  # noqa
        assert con_matrix.matrix_proportions['Total'].values.tolist() == [0.5938375350140056, 0.4061624649859944, 1.0]  # noqa
        assert con_matrix.matrix_proportions.index.values.tolist() == [0, 1, 'Total']
        assert con_matrix.matrix_proportions.columns.values.tolist() == [0, 1, 'Total']

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
        con_matrix = TwoClassConfusionMatrix(actual_classes=mock_data.actual,
                                             predicted_classes=mock_data.predictions,
                                             positive_class=1)
        self.check_confusion_matrix(con_matrix, mock_data)

        evaluator = TwoClassEvaluator(positive_class=1)
        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=mock_data.predictions)
        assert isclose(con_matrix.all_quality_metrics['Kappa'], evaluator.all_quality_metrics['Kappa'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['F1 Score'], evaluator.all_quality_metrics['F1 Score'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['Two-Class Accuracy'], evaluator.all_quality_metrics['Two-Class Accuracy'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['Error Rate'], evaluator.all_quality_metrics['Error Rate'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['True Positive Rate'], evaluator.all_quality_metrics['True Positive Rate'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['True Negative Rate'], evaluator.all_quality_metrics['True Negative Rate'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['False Positive Rate'], evaluator.all_quality_metrics['False Positive Rate'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['False Negative Rate'], evaluator.all_quality_metrics['False Negative Rate'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['Positive Predictive Value'], evaluator.all_quality_metrics['Positive Predictive Value'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['Negative Predictive Value'], evaluator.all_quality_metrics['Negative Predictive Value'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['Prevalence'], evaluator.all_quality_metrics['Prevalence'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['No Information Rate'], evaluator.all_quality_metrics['No Information Rate'])  # noqa
        assert isclose(con_matrix.all_quality_metrics['Total Observations'], evaluator.all_quality_metrics['Total Observations'])  # noqa

        ######################################################################################################
        # `from_classes` check calculations *******SWAPPED********
        ######################################################################################################
        con_matrix = TwoClassConfusionMatrix(actual_classes=mock_data.actual,
                                             predicted_classes=mock_data.predictions,
                                             positive_class=0)

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

        evaluator = TwoClassProbabilityEvaluator(converter=TwoClassThresholdConverter(threshold=0.5, positive_class=1))  # noqa
        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=mock_data[['pos_probabilities', 'neg_probabilities']].rename(columns={'pos_probabilities': 1, 'neg_probabilities': 0}))  # noqa
        self.check_confusion_matrix(con_matrix=evaluator._confusion_matrix, mock_data=mock_data)

        ######################################################################################################
        # try a threshold of 1, which means that 0 positives will be predicted
        ######################################################################################################
        evaluator = TwoClassProbabilityEvaluator(converter=TwoClassThresholdConverter(threshold=1, positive_class=1))  # noqa
        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=mock_data[['pos_probabilities', 'neg_probabilities']].rename(columns={'pos_probabilities': 1, 'neg_probabilities': 0}))  # noqa

        assert evaluator._confusion_matrix.matrix.loc[:, 0].values.tolist() == [424, 290, 714]
        assert evaluator._confusion_matrix.matrix.loc[:, 1].values.tolist() == [0, 0, 0]
        assert evaluator._confusion_matrix.matrix.loc[:, 'Total'].values.tolist() == [424, 290, 714]
        assert evaluator._confusion_matrix.matrix.index.values.tolist() == [0, 1, 'Total']
        assert evaluator._confusion_matrix.matrix.columns.values.tolist() == [0, 1, 'Total']
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['Kappa'], 0)
        assert evaluator._confusion_matrix.all_quality_metrics['F1 Score'] is None
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['Two-Class Accuracy'], evaluator._confusion_matrix.negative_predictive_value)  # noqa
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['Error Rate'], evaluator._confusion_matrix.prevalence)  # noqa
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['True Positive Rate'], 0)
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['True Negative Rate'], 1)
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['False Positive Rate'], 0)
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['False Negative Rate'], 1)
        assert evaluator._confusion_matrix.all_quality_metrics['Positive Predictive Value'] is None
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['Negative Predictive Value'], evaluator._confusion_matrix.all_quality_metrics['No Information Rate'])  # noqa
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['Prevalence'], 0.4061624649859944)
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['No Information Rate'], 0.5938375350140056)  # noqa
        assert isclose(evaluator._confusion_matrix.all_quality_metrics['Total Observations'], len(mock_data))

    def test_TwoClassEvaluator_from_classes(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        evaluator = TwoClassEvaluator.from_classes(actual_classes=mock_data.actual, predicted_classes=mock_data.predictions, positive_class=1)  # noqa
        self.check_confusion_matrix(con_matrix=evaluator._confusion_matrix, mock_data=mock_data)

    def test_TwoClassEvaluator_probabilities_custom_threshold(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa

        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        evaluator = TwoClassProbabilityEvaluator(converter=TwoClassThresholdConverter(positive_class=1, threshold=0.5))  # noqa
        evaluator.evaluate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(evaluator.auc, roc_auc_score(y_true=mock_data.actual, y_score=predictions_mock[1]))
        assert isclose(evaluator.auc, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa
        self.check_confusion_matrix(con_matrix=evaluator.confusion_matrix, mock_data=mock_data)

        # test ROC calculations
        converter = TwoClassRocOptimizerConverter(actual_classes=mock_data.actual, positive_class=1)
        converter.convert(values=predictions_mock)
        actual_thresholds = converter.false_positive_rates, converter.true_positive_rates, converter.ideal_threshold  # noqa
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/fpr_tpr_threshold_mock.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(actual_thresholds, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_thresholds = pickle.load(saved_object)
            assert len(expected_thresholds) == 3
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[0], actual_thresholds[0])])
            assert all([isclose(x, y) for x, y in zip(expected_thresholds[1], actual_thresholds[1])])
            assert isclose(expected_thresholds[2], actual_thresholds[2])

        # test PPV/TPR calculations
        converter = TwoClassPrecisionRecallOptimizerConverter(actual_classes=mock_data.actual, positive_class=1)  # noqa
        converter.convert(values=predictions_mock)
        actual_thresholds = converter.positive_predictive_values, converter.true_positive_rates, converter.ideal_threshold  # noqa
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

    # noinspection SpellCheckingInspection
    # noinspection PyTypeChecker
    def test_ConfusionMatrix_MultiClass(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_MultiClass_predictions.csv')))  # noqa
        evaluator = MultiClassEvaluator.from_classes(actual_classes=mock_data.actual, predicted_classes=mock_data.predicted_classes)  # noqa

        assert evaluator.confusion_matrix.total_observations == 38
        assert evaluator.total_observations == 38
        assert evaluator.matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert evaluator.matrix['versicolor'].values.tolist() == [0, 12, 2, 14]
        assert evaluator.matrix['virginica'].values.tolist() == [0, 1, 11, 12]
        assert evaluator.matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert evaluator.matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert evaluator.matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

        ######################################################################################################
        # change all setosa predictions to versicolor
        # i.e. there will be no predictions for a class (setosa), make sure Confusion Matrix can handle that.
        ######################################################################################################
        no_setosa = np.array([x if x != 'setosa' else 'versicolor' for x in mock_data.predicted_classes])
        evaluator = MultiClassEvaluator.from_classes(actual_classes=mock_data.actual, predicted_classes=no_setosa)  # noqa

        assert evaluator.confusion_matrix.total_observations == 38
        assert evaluator.total_observations == 38
        assert evaluator.matrix['setosa'].values.tolist() == [0, 0, 0, 0]
        assert evaluator.matrix['versicolor'].values.tolist() == [12, 12, 2, 26]
        assert evaluator.matrix['virginica'].values.tolist() == [0, 1, 11, 12]
        assert evaluator.matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert evaluator.matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert evaluator.matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

        ######################################################################################################
        # test from probabilities
        ######################################################################################################
        evaluator = MultiClassEvaluator(converter=HighestValueConverter())
        evaluator.evaluate(actual_values=mock_data.actual,
                           predicted_values=mock_data[['setosa', 'versicolor', 'virginica']])
        assert evaluator.matrix['setosa'].values.tolist() == [12, 0, 0, 12]
        assert evaluator.matrix['versicolor'].values.tolist() == [0, 12, 2, 14]
        assert evaluator.matrix['virginica'].values.tolist() == [0, 1, 11, 12]
        assert evaluator.matrix['Total'].values.tolist() == [12, 13, 13, 38]
        assert evaluator.matrix.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']
        assert evaluator.matrix.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']

        assert evaluator.confusion_matrix.matrix_proportions['setosa'].values.tolist() == [0.3157894736842105, 0.0, 0.0, 0.3157894736842105]  # noqa
        assert evaluator.confusion_matrix.matrix_proportions['versicolor'].values.tolist() == [0.0, 0.3157894736842105, 0.05263157894736842, 0.3684210526315789]  # noqa
        assert evaluator.confusion_matrix.matrix_proportions['virginica'].values.tolist() == [0.0, 0.02631578947368421, 0.2894736842105263, 0.3157894736842105]  # noqa
        assert evaluator.confusion_matrix.matrix_proportions['Total'].values.tolist() == [0.3157894736842105, 0.34210526315789475, 0.34210526315789475, 1.0]  # noqa
        assert evaluator.confusion_matrix.matrix_proportions.index.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']  # noqa
        assert evaluator.confusion_matrix.matrix_proportions.columns.values.tolist() == ['setosa', 'versicolor', 'virginica', 'Total']  # noqa

        TestHelper.check_plot('data/test_Evaluators/test_confusion_matrix_heatmap_no_totals.png',
                              lambda: evaluator.confusion_matrix.get_heatmap(include_totals=False))
        TestHelper.check_plot('data/test_Evaluators/test_confusion_matrix_heatmap_with_totals.png',
                              lambda: evaluator.confusion_matrix.get_heatmap(include_totals=True))

    # noinspection SpellCheckingInspection
    def test_ConfusionMatrix_MultiClass_scores(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_MultiClass_predictions.csv')))  # noqa

        evaluator = MultiClassEvaluator(converter=HighestValueConverter())
        evaluator.evaluate(actual_values=mock_data.actual,
                           predicted_values=mock_data[['setosa', 'versicolor', 'virginica']])

        assert isclose(evaluator.accuracy, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predicted_classes))  # noqa
        assert isclose(evaluator.all_quality_metrics['Kappa'], cohen_kappa_score(y1=mock_data.actual, y2=mock_data.predicted_classes))  # noqa
        assert isclose(evaluator.all_quality_metrics['Accuracy'], accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predicted_classes))  # noqa
        assert isclose(evaluator.all_quality_metrics['Error Rate'], 1 - accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predicted_classes))  # noqa
        assert isclose(evaluator.all_quality_metrics['No Information Rate'], 0.34210526315789475)
        assert isclose(evaluator.all_quality_metrics['Total Observations'], 38)

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/mock_metrics_per_class.pkl'))  # noqa
        # with open(file, 'wb') as output:
        #     pickle.dump(con_matrix.metrics_per_class, output, pickle.HIGHEST_PROTOCOL)
        with open(file, 'rb') as saved_object:
            expected_metrics_per_class = pickle.load(saved_object)
            assert TestHelper.ensure_all_values_equal(data_frame1=expected_metrics_per_class,
                                                      data_frame2=evaluator.metrics_per_class)
