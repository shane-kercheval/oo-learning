from oolearning.evaluators.EvaluatorBase import EvaluatorBase

import numpy as np


class RegressionEvaluator(EvaluatorBase):
    def __init__(self):
        self._residuals = None
        self._standard_deviation = None
        self._mean_squared_error = None
        self._mean_absolute_error = None

    def evaluate(self, actual_values: np.ndarray, predicted_values: object):
        self._residuals = predicted_values - actual_values
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
    def all_quality_metrics(self) -> dict:
        return {'Mean Absolute Error (MAE)': self.mean_absolute_error,
                'Mean Squared Error (MSE)': self.mean_squared_error,
                'Root Mean Squared Error (RMSE)': self.root_mean_squared_error,
                'RMSE to Standard Deviation of Target': self.rmse_to_st_dev}
