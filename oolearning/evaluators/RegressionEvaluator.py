import numpy as np
from matplotlib import pyplot as plt

from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.evaluators.MaeScore import MaeScore
from oolearning.evaluators.MseScore import MseScore
from oolearning.evaluators.RSquaredScore import RSquaredScore

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from statsmodels import api as sm  # https://github.com/statsmodels/statsmodels/issues/3814


# noinspection SpellCheckingInspection
class RegressionEvaluator(EvaluatorBase):
    """
    Evaluates models for regresion (i.e. numeric outcome) problems.
    """
    def __init__(self):
        self._actual_values = None
        self._predicted_values = None
        self._residuals = None
        self._standard_deviation = None
        self._mean_squared_error = None
        self._mean_absolute_error = None
        self._r_squared = None

    def evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray):
        assert len(actual_values) == len(predicted_values)
        self._actual_values = actual_values
        self._predicted_values = predicted_values
        self._residuals = actual_values - predicted_values
        self._standard_deviation = np.std(actual_values)
        self._mean_squared_error = MseScore().calculate(actual_values=actual_values,
                                                        predicted_values=predicted_values)
        self._mean_absolute_error = MaeScore().calculate(actual_values=actual_values,
                                                         predicted_values=predicted_values)
        self._r_squared = RSquaredScore().calculate(actual_values=actual_values,
                                                    predicted_values=predicted_values)
        return self

    @property
    def mean_absolute_error(self) -> float:
        return self._mean_absolute_error

    @property
    def mean_squared_error(self) -> float:
        return self._mean_squared_error

    @property
    def root_mean_squared_error(self) -> float:
        return np.sqrt(self.mean_squared_error)

    @property
    def rmse_to_st_dev(self) -> float:
        return self.root_mean_squared_error / self._standard_deviation

    @property
    def r_squared(self) -> float:
        return self._r_squared

    @property
    def total_observations(self):
        return len(self._actual_values)

    @property
    def all_quality_metrics(self) -> dict:
        return {'Mean Absolute Error (MAE)': self.mean_absolute_error,
                'Mean Squared Error (MSE)': self.mean_squared_error,
                'Root Mean Squared Error (RMSE)': self.root_mean_squared_error,
                'RMSE to Standard Deviation of Target': self.rmse_to_st_dev,
                'R Squared': self.r_squared,
                'Total Observations': self.total_observations}

    def plot_residuals_vs_fits(self):
        lowess = sm.nonparametric.lowess
        loess_points = lowess(self._residuals, self._predicted_values)
        loess_x, loess_y = zip(*loess_points)

        plt.plot(loess_x, loess_y, color='r')
        plt.scatter(x=self._predicted_values, y=self._residuals, s=8, alpha=0.5)
        plt.title('Residuals vs. Fitted Values')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals (Actual - Predicted)')
        return plt.gca()

    def plot_predictions_vs_actuals(self):
        lowess = sm.nonparametric.lowess
        loess_points = lowess(self._predicted_values, self._actual_values)
        loess_x, loess_y = zip(*loess_points)

        plt.plot(loess_x, loess_y, color='r', alpha=0.5, label='Loess (Predictions vs Actuals)')
        plt.plot(self._actual_values, self._actual_values, color='b', alpha=0.5, label='Perfect Prediction')
        plt.scatter(x=self._actual_values, y=self._predicted_values, s=8, alpha=0.5)
        plt.title('Predicted Values vs. Actual Values')
        plt.xlabel('Actuals')
        plt.ylabel('Predicted')
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels)
        plt.figtext(0.99, 0.01,
                    'Note: observations above blue line mean model is over-predicting; below means under-predicting.',  # noqa
                    horizontalalignment='right')
        return ax

    def plot_residuals_vs_actuals(self):
        lowess = sm.nonparametric.lowess
        loess_points = lowess(self._residuals, self._actual_values)
        loess_x, loess_y = zip(*loess_points)

        plt.plot(loess_x, loess_y, color='r')
        plt.scatter(x=self._actual_values, y=self._residuals, s=8, alpha=0.5)
        plt.title('Residuals vs. Actual Values')
        plt.xlabel('Actual')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.figtext(0.99, 0.01,
                    'Note: Actual > Predicted => Under-predicting (positive residual); negative residuals mean over-predicting',  # noqa
                    horizontalalignment='right')
