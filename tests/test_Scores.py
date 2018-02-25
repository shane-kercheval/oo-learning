import os
from math import isclose

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, cohen_kappa_score, \
    mean_squared_error, mean_absolute_error, roc_auc_score

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic
class EvaluatorTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_BaseClass(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        rmse_eval = RmseScore()
        accuracy = rmse_eval.calculate(actual_values=actual, predicted_values=predicted)
        assert isclose(accuracy, 2.9154759474226504)
        assert isclose(rmse_eval.value, 2.9154759474226504)
        # should not be able to call calculate twice on same object (could indicate some sort of reuse error)
        self.assertRaises(AssertionError,
                          lambda: rmse_eval.calculate(actual_values=actual, predicted_values=predicted))

        assert isinstance(rmse_eval, CostFunctionMixin)

    def test_RmseScore(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        rmse_eval = RmseScore()
        assert isinstance(rmse_eval, CostFunctionMixin)
        assert isinstance(rmse_eval, ScoreBase)
        assert rmse_eval.name == Metric.ROOT_MEAN_SQUARE_ERROR.value
        rmse_eval.calculate(actual_values=actual, predicted_values=predicted)
        assert isclose(np.sqrt(mean_squared_error(y_true=actual, y_pred=predicted)), rmse_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        rmse_other = RmseScore()
        rmse_other.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(rmse_other.value, 3.5355339059327378)  # "worse"
        eval_list = [rmse_other, rmse_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [3.5355339059327378, 2.9154759474226504])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [2.9154759474226504, 3.5355339059327378])])

    def test_MaeScore(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        mae_eval = MaeScore()
        assert isinstance(mae_eval, CostFunctionMixin)
        assert isinstance(mae_eval, ScoreBase)
        assert mae_eval.name == Metric.MEAN_ABSOLUTE_ERROR.value
        mae_eval.calculate(actual_values=actual, predicted_values=predicted)
        assert isclose(mean_absolute_error(y_true=actual, y_pred=predicted), mae_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        rmse_other = RmseScore()
        rmse_other.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(rmse_other.value, 3.5355339059327378)  # "worse"

        eval_list = [rmse_other, mae_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [3.5355339059327378, 2.3333333333333335])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [2.3333333333333335, 3.5355339059327378])])

    def test_AucScore(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        score = AucScore(positive_class=1)
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)

        score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score.value, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa
        ######################################################################################################
        # Test sorting
        ######################################################################################################
        # makes a 'worse
        score_other = AucScore(positive_class=0)
        score_other.calculate(actual_values=np.array([1 if x == 0 else 0 for x in mock_data.actual]),
                              predicted_values=predictions_mock)
        assert isclose(score_other.value, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.neg_probabilities))  # noqa

        score_list = [score_other, score]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in score_list],
                                                  [0.25571324007807417, 0.74428675992192583])])
        score_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in score_list],
                                                  [0.74428675992192583, 0.25571324007807417])])

    def test_KappaScore(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        score = KappaScore(converter=TwoClassThresholdConverter(threshold=0.41, positive_class=1))
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)

        accuracy = score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.37990215607221967)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        # creates worse value
        score_other = KappaScore(converter=HighestValueConverter())  # same as threshold of 0.5
        # score_other = KappaScore(converter=TwoClassThresholdConverter(threshold=0.5))
        score_other.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score_other.value, cohen_kappa_score(y1=mock_data.actual, y2=mock_data.predictions))

        eval_list = [score_other, score]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.34756903797404387, 0.37990215607221967])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.37990215607221967, 0.34756903797404387])])

    def test_F1Score(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        score = F1Score(converter=TwoClassThresholdConverter(threshold=0.41, positive_class=1))
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)
        score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score.value, 0.6472491909385113)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        score_other = F1Score(converter=TwoClassThresholdConverter(threshold=0.5,
                                                                   positive_class=1))
        score_other.calculate(actual_values=mock_data.actual,
                              predicted_values=predictions_mock)
        assert isclose(score_other.value, f1_score(y_true=mock_data.actual, y_pred=mock_data.predictions,  pos_label=1))  # noqa

        eval_list = [score_other, score]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.5802707930367504, 0.6472491909385113])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.6472491909385113, 0.5802707930367504])])

    def test_ErrorRate(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        score = ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.41, positive_class=1))
        assert isinstance(score, CostFunctionMixin)
        assert isinstance(score, ScoreBase)
        score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score.value, 0.30532212885154064)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        score_other = ErrorRateScore(converter=TwoClassThresholdConverter(threshold=0.5,
                                                                          positive_class=1))
        score_other.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score_other.value, 1 - 0.696078431372549)

        eval_list = [score, score_other]  # "worse, better"
        # lower error is better
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.30532212885154064, 0.303921568627451])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.303921568627451, 0.30532212885154064])])

    def test_Misc_scores(self):
        """
        For example, these holdout_scores might be already tested in another class (e.g. Sensitivity is
            tested via TwoClassEvaluator), but we want to verify we can instantiate and use.
        """
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]
        ######################################################################################################
        score = SensitivityScore(converter=TwoClassThresholdConverter(positive_class=1, threshold=0.5))
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)
        accuracy = score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions))
        assert isclose(score.value, recall_score(y_true=mock_data.actual, y_pred=mock_data.predictions))
        ######################################################################################################
        score = SpecificityScore(converter=TwoClassThresholdConverter(positive_class=1, threshold=0.5))
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)
        accuracy = score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.8183962264150944)
        assert isclose(score.value, 0.8183962264150944)
        ######################################################################################################
        score = AccuracyScore(converter=TwoClassThresholdConverter(positive_class=1, threshold=0.5))
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)
        accuracy = score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))
        assert isclose(score.value, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predictions))

    def test_KappaScore_multi_class(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_MultiClass_predictions.csv')))  # noqa

        score = KappaScore(converter=HighestValueConverter())
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)

        # noinspection SpellCheckingInspection
        score.calculate(actual_values=mock_data.actual,
                        predicted_values=mock_data[['setosa', 'versicolor', 'virginica']])
        assert isclose(score.value, cohen_kappa_score(y1=mock_data.actual, y2=mock_data.predicted_classes))

    def test_Accuracy_multi_class(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_MultiClass_predictions.csv')))  # noqa

        score = AccuracyScore(converter=HighestValueConverter())
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)

        # noinspection SpellCheckingInspection
        score.calculate(actual_values=mock_data.actual,
                        predicted_values=mock_data[['setosa', 'versicolor', 'virginica']])
        assert isclose(score.value, accuracy_score(y_true=mock_data.actual, y_pred=mock_data.predicted_classes))  # noqa
