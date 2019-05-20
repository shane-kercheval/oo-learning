import os
from math import isclose
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, fbeta_score, cohen_kappa_score, \
    mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score

from oolearning import *
from oolearning.model_wrappers.ModelExceptions import AlreadyExecutedError
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic
class ScoreTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_ScoreMediator(self):
        ######################################################################################################
        # test ScoreMediator with a ScoreActualPredictedBase object
        ######################################################################################################
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        score = KappaScore(converter=TwoClassThresholdConverter(threshold=0.41, positive_class=1))

        # check that both the score is returned from the Mediator and the score object has the `value` set
        accuracy = ScoreMediator.calculate(score,
                                           data_x=None,
                                           actual_target_variables=mock_data.actual,
                                           predicted_values=predictions_mock)
        assert isclose(accuracy, 0.37990215607221967)  # check the score is returned
        assert isclose(score.value, 0.37990215607221967)  # check the score object's `value` is set

        ######################################################################################################
        # test ScoreMediator with a ScoreClusteringBase object
        ######################################################################################################
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        trainer = ModelTrainer(model=ClusteringHierarchical(),
                               model_transformations=[NormalizationVectorSpaceTransformer()],
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringHierarchicalHP(num_clusters=3))

        score = SilhouetteScore()
        assert score.name == Metric.SILHOUETTE.value
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreClusteringBase)

        accuracy = ScoreMediator.calculate(score,
                                           # NOTE: we have to pass in the TRANSFORMED data
                                           data_x=NormalizationVectorSpaceTransformer().fit_transform(cluster_data),  # noqa
                                           actual_target_variables=None,
                                           predicted_values=clusters)
        assert isclose(accuracy, 0.5562322357473719)  # check the score is returned
        assert isclose(score.value, 0.5562322357473719)  # check the score object's `value` is set

        ######################################################################################################
        # test ScoreMediator with unsupported ScoreBaseObject
        ######################################################################################################
        class MockScore(ScoreBase):
            def _better_than(self, this: float, other: float) -> bool:
                pass

            def _calculate(self, *args) -> float:
                pass

            # noinspection PyPropertyDefinition
            @property
            def name(self) -> str:
                pass

        self.assertRaises(ValueError,
                          lambda: ScoreMediator.calculate(MockScore(),
                                                          data_x=None,
                                                          actual_target_variables=None,
                                                          predicted_values=None))

    def test_BaseClass(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        rmse_eval = RmseScore()
        accuracy = rmse_eval.calculate(actual_values=actual, predicted_values=predicted)
        assert isclose(accuracy, 2.9154759474226504)
        assert isclose(rmse_eval.value, 2.9154759474226504)
        # should not be able to call calculate twice on same object (could indicate some sort of reuse error)
        self.assertRaises(AlreadyExecutedError,
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

    def test_RmsleScore(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        expected_score = np.sqrt(np.mean((np.log(1+actual) - np.log(1+predicted))**2))
        rmse_eval = RmsleScore()
        assert isinstance(rmse_eval, CostFunctionMixin)
        assert isinstance(rmse_eval, ScoreBase)
        assert rmse_eval.name == Metric.ROOT_MEAN_SQUARE_LOGARITHMIC_ERROR.value
        rmse_eval.calculate(actual_values=actual, predicted_values=predicted)
        assert isclose(expected_score, rmse_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        rmse_other = RmsleScore()
        rmse_other.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        expected_other_score = 0.42204286369153776
        assert isclose(rmse_other.value, expected_other_score)  # "worse"
        eval_list = [rmse_other, rmse_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [expected_other_score, expected_score])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [expected_score, expected_other_score])])

    def test_MseScore(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        mse_eval = MseScore()
        assert isinstance(mse_eval, CostFunctionMixin)
        assert isinstance(mse_eval, ScoreBase)
        assert mse_eval.name == Metric.MEAN_SQUARED_ERROR.value
        score = mse_eval.calculate(actual_values=actual, predicted_values=predicted)

        assert score == 8.5
        assert isclose(score, RmseScore().calculate(actual_values=actual, predicted_values=predicted) ** 2)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        mse_other = MseScore()
        mse_other.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(mse_other.value, 12.5)  # "worse"
        eval_list = [mse_other, mse_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [12.5, 8.5])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [8.5, 12.5])])

    def test_MspeScore(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        expected_score = float(np.mean(np.square((actual - predicted) / actual)))
        mspe_eval = MspeScore()
        assert isinstance(mspe_eval, CostFunctionMixin)
        assert isinstance(mspe_eval, ScoreBase)
        assert mspe_eval.name == Metric.MEAN_SQUARED_PERCENTAGE_ERROR.value
        score = mspe_eval.calculate(actual_values=actual, predicted_values=predicted)

        # need to round because of the small noise we get from adding a constant to help avoid divide-by-zero
        assert isclose(score, expected_score)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        mspe_other = MspeScore()
        mspe_other.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(mspe_other.value, 0.4770842603697943)  # "worse"
        eval_list = [mspe_other, mspe_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.4770842603697943, 0.12248815407984363])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.12248815407984363, 0.4770842603697943])])

        ######################################################################################################
        # make sure we don't get a divide by zero
        ######################################################################################################
        predicted = np.array([0, 11, 0, 11, 0.1, 0, 7, 8, 11, 13, 0.5, 0])
        actual = np.array([0, 11, 1, 16, 0, 0.3, 5, 13, 12, 13, 1, 0.24])
        constant = 1
        expected_score = float(np.mean(np.square(((actual + constant) - (predicted + constant)) / (actual + constant))))
        mspe_eval = MspeScore(constant=constant)
        score = mspe_eval.calculate(actual_values=actual, predicted_values=predicted)

        assert isclose(expected_score, score)

    def test_R_Squared_Score(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        r2_eval = RSquaredScore()
        assert isinstance(r2_eval, UtilityFunctionMixin)
        assert isinstance(r2_eval, ScoreBase)
        assert r2_eval.name == Metric.R_SQUARED.value
        r2_eval.calculate(actual_values=actual, predicted_values=predicted)
        assert isclose(0.41714285714285715, r2_eval.value)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        r2_worse = RSquaredScore()
        r2_worse.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(r2_worse.value, 0.1428571428571429)  # "worse"
        eval_list = [r2_worse, r2_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.1428571428571429, 0.41714285714285715])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.41714285714285715, 0.1428571428571429])])

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
        mae_other = MaeScore()
        mae_other.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(mae_other.value, 3.1666666666666665)  # "worse"

        eval_list = [mae_other, mae_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [3.1666666666666665, 2.3333333333333335])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [2.3333333333333335, 3.1666666666666665])])

    def test_MapeScore(self):
        predicted = np.array([7, 10, 12, 10, 10, 8, 7, 8, 11, 13, 10, 8])
        actual = np.array([6, 10, 14, 16, 7, 5, 5, 13, 12, 13, 8, 5])
        expected_score = float(np.mean(np.abs((actual - predicted) / actual)))
        mspe_eval = MapeScore()
        assert isinstance(mspe_eval, CostFunctionMixin)
        assert isinstance(mspe_eval, ScoreBase)
        assert mspe_eval.name == Metric.MEAN_ABSOLUTE_PERCENTAGE_ERROR.value
        score = mspe_eval.calculate(actual_values=actual, predicted_values=predicted)

        # need to round because of the small noise we get from adding a constant to help avoid divide-by-zero
        assert isclose(score, expected_score)

        ######################################################################################################
        # Test sorting
        ######################################################################################################
        mspe_other = MapeScore()
        mspe_other.calculate(actual_values=actual - 1, predicted_values=predicted + 1)  # create more spread
        assert isclose(mspe_other.value, 0.5417688792688793)  # "worse"
        eval_list = [mspe_other, mspe_eval]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.5417688792688793, 0.2859203296703297])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.2859203296703297, 0.5417688792688793])])
        ######################################################################################################
        # make sure we don't get a divide by zero
        ######################################################################################################
        predicted = np.array([0, 11, 0, 11, 0.1, 0, 7, 8, 11, 13, 0.5, 0])
        actual = np.array([0, 11, 1, 16, 0, 0.3, 5, 13, 12, 13, 1, 0.24])
        constant = 1
        expected_score = float(np.mean(np.abs(((actual + constant) - (predicted + constant)) / (actual + constant))))
        mspe_eval = MapeScore(constant=constant)
        score = mspe_eval.calculate(actual_values=actual, predicted_values=predicted)

        assert isclose(expected_score, score)

    def test_AucROCScore(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        score = AucRocScore(positive_class=1)
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)

        score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score.value, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa
        ######################################################################################################
        # Test sorting
        ######################################################################################################
        # makes a 'worse
        score_other = AucRocScore(positive_class=0)
        score_other.calculate(actual_values=np.array([1 if x == 0 else 0 for x in mock_data.actual]),
                              predicted_values=predictions_mock)
        assert isclose(score_other.value, roc_auc_score(y_true=mock_data.actual, y_score=mock_data.neg_probabilities))  # noqa

        score_list = [score_other, score]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in score_list],
                                                  [0.25571324007807417, 0.74428675992192583])])
        score_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in score_list],
                                                  [0.74428675992192583, 0.25571324007807417])])

    def test_AucPrecisionRecallScore(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        score = AucPrecisionRecallScore(positive_class=1)
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)

        score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score.value, average_precision_score(y_true=mock_data.actual, y_score=mock_data.pos_probabilities))  # noqa
        ######################################################################################################
        # Test sorting
        ######################################################################################################
        # makes a 'worse
        score_other = AucPrecisionRecallScore(positive_class=0)
        score_other.calculate(actual_values=np.array([1 if x == 0 else 0 for x in mock_data.actual]),
                              predicted_values=predictions_mock)
        assert isclose(score_other.value, average_precision_score(y_true=mock_data.actual, y_score=mock_data.neg_probabilities))  # noqa

        score_list = [score_other, score]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in score_list],
                                                  [0.28581244853623045, 0.6659419996895501])])
        score_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in score_list],
                                                  [0.6659419996895501, 0.28581244853623045])])

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

    def test_FBetaScore(self):
        mock_data = pd.read_csv(os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_Evaluators/test_ConfusionMatrix_mock_actual_predictions.csv')))  # noqa
        predictions_mock = mock_data.drop(columns=['actual', 'predictions'])
        predictions_mock.columns = [1, 0]

        ######################################################################################################
        # F1 Score (i.e. Beta == 1)
        ######################################################################################################
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

        ######################################################################################################
        # FBeta Score (Beta == 0.5)
        ######################################################################################################
        score_other = FBetaScore(converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1),
                                 beta=0.5)
        score_other.calculate(actual_values=mock_data.actual,
                              predicted_values=predictions_mock)
        assert isclose(score_other.value,
                       fbeta_score(y_true=mock_data.actual,
                                   y_pred=mock_data.predictions,
                                   beta=0.5,
                                   pos_label=1))

        eval_list = [score_other, score]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.6260434056761269, 0.6472491909385113])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.6472491909385113, 0.6260434056761269])])

        ######################################################################################################
        # FBeta Score (Beta == 1.5)
        ######################################################################################################
        score_other = FBetaScore(converter=TwoClassThresholdConverter(threshold=0.5,
                                                                      positive_class=1),
                                 beta=1.5)
        score_other.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(score_other.value,
                       fbeta_score(y_true=mock_data.actual,
                                   y_pred=mock_data.predictions,
                                   beta=1.5,
                                   pos_label=1))

        eval_list = [score_other, score]  # "worse, better"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.5542922114837977, 0.6472491909385113])])
        eval_list.sort()  # "better, worse"
        assert all([isclose(x, y) for x, y in zip([x.value for x in eval_list],
                                                  [0.6472491909385113, 0.5542922114837977])])

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

    def test_BaseValue_is_int_or_float(self):
        # bug, where positive predictive value (or any score) returns 0 (e.g. from DummyClassifier)
        # which is an int (but base class originally checked only for float)
        class MockScore(ScoreActualPredictedBase):
            @property
            def name(self) -> str:
                return 'test'

            def _better_than(self, this: float, other: float) -> bool:
                return False

            def _calculate(self, actual_values: np.ndarray, predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:  # noqa
                return 0

        score = MockScore()
        # ensure .calculate doesn't explode
        # noinspection PyTypeChecker
        score.calculate(actual_values=[], predicted_values=[])

    def test_Misc_scores(self):
        """
        For example, these holdout_score_objects might be already tested in another class (e.g. Sensitivity is
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
        score = PositivePredictiveValueScore(converter=TwoClassThresholdConverter(positive_class=1, threshold=0.5))  # noqa
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)
        accuracy = score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.6607929515418502)
        assert isclose(score.value, 0.6607929515418502)
        ######################################################################################################
        score = NegativePredictiveValueScore(converter=TwoClassThresholdConverter(positive_class=1, threshold=0.5))  # noqa
        assert isinstance(score, UtilityFunctionMixin)
        assert isinstance(score, ScoreBase)
        accuracy = score.calculate(actual_values=mock_data.actual, predicted_values=predictions_mock)
        assert isclose(accuracy, 0.7125256673511293)
        assert isclose(score.value, 0.7125256673511293)
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
