import numpy as np
from matplotlib import pyplot as plt
from statsmodels import api as sm

from oolearning.evaluators.EvaluatorBase import EvaluatorBase


# noinspection SpellCheckingInspection
class RegressionEvaluator(EvaluatorBase):
    def __init__(self):
        self._actual_values = None
        self._predicted_values = None
        self._residuals = None
        self._standard_deviation = None
        self._mean_squared_error = None
        self._mean_absolute_error = None

    def evaluate(self, actual_values: np.ndarray, predicted_values: object):
        assert len(actual_values) == len(predicted_values)
        self._actual_values = actual_values
        self._predicted_values = predicted_values
        self._residuals = actual_values - predicted_values
        self._standard_deviation = np.std(actual_values)
        self._mean_squared_error = np.mean(np.square(self._residuals))
        self._mean_absolute_error = np.mean(np.abs(self._residuals))

        return self

    @property
    def mean_absolute_error(self):
        return self._mean_absolute_error

    @property
    def mean_squared_error(self):
        return self._mean_squared_error

    @property
    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error)

    @property
    def rmse_to_st_dev(self):
        return self.root_mean_squared_error / self._standard_deviation

    @property
    def total_observations(self):
        return len(self._actual_values)

    @property
    def all_quality_metrics(self) -> dict:
        return {'Mean Absolute Error (MAE)': self.mean_absolute_error,
                'Mean Squared Error (MSE)': self.mean_squared_error,
                'Root Mean Squared Error (RMSE)': self.root_mean_squared_error,
                'RMSE to Standard Deviation of Target': self.rmse_to_st_dev,
                'Total Observations': self.total_observations}

    def plot_residuals_vs_fits(self):
        from statsmodels import api as sm
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
