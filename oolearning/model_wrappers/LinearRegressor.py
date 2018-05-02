from collections import OrderedDict

import numpy as np
import pandas as pd
from matplotlib import figure, pyplot as plt

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    from statsmodels import api as sm  # https://github.com/statsmodels/statsmodels/issues/3814

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase  # noqa
from oolearning.model_wrappers.ModelExceptions import MissingValueError  # noqa
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase  # noqa


class LinearRegressor(ModelWrapperBase):
    """
    fits Linear Regression model on the data
    """

    def __init__(self):
        super().__init__()
        self._training_target_std = None

    @staticmethod
    def _get_significance_code(p_value):
        """
        Significance codes:
            ‘***’ if <= 0.001
            ‘**’  if <= 0.01
            ‘*’   if <= 0.05
            ‘.’   if <= 0.1
            else ‘ ’
        :param p_value: p-value to convert
        :return: code
        """
        if p_value <= 0.001:
            return '***'
        elif p_value <= 0.01:
            return '**'
        elif p_value <= 0.05:
            return '*'
        elif p_value <= 0.10:
            return '.'
        else:
            return ''

    # noinspection SpellCheckingInspection
    @property
    def graph(self) -> figure.Figure:
        """
        traditional graphs for regression curve
        :return:
        """
        # TODO: finish plots, finish doc comments
        # https://medium.com/@emredjan/emulating-r-regression-plots-in-python-43741952c034
        # fitted values (need a constant term for intercept)
        model_fitted_y = self._model_object.fittedvalues
        # model residuals
        model_residuals = self._model_object.resid.values
        # normalized residuals
        model_norm_residuals = self._model_object.get_influence().resid_studentized_internal
        # absolute squared normalized residuals
        # model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
        # absolute residuals
        # model_abs_resid = np.abs(model_residuals)
        # leverage, from statsmodels internals
        model_leverage = self._model_object.get_influence().hat_matrix_diag

        # cook's distance, from stats models internals
        model_cooks = self._model_object.get_influence().cooks_distance[0]

        lowess = sm.nonparametric.lowess
        loess_points = lowess(model_residuals, model_fitted_y)
        loess_x, loess_y = zip(*loess_points)

        fig = plt.figure(figsize=(11, 7))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(loess_x, loess_y, color='r')
        ax1.scatter(x=model_fitted_y, y=model_residuals, s=8, alpha=0.5)
        ax1.set(**{'title': 'Residuals vs. Fitted Values',
                   'xlabel': 'Fitted Values',
                   'ylabel': 'Residuals'})

        ax2 = fig.add_subplot(2, 2, 2)
        # http://scientificpythonsnippets.com/index.php/2-uncategorised/6-q-q-plot-in-python-to-test-if-data-is-normally-distributed
        data = model_norm_residuals
        data.sort()
        np.random.seed(42)
        norm = np.random.normal(0, 1, len(data))
        norm.sort()
        ax2.scatter(norm, data, s=8, alpha=0.5)
        # generate a trend line as in http://widu.tumblr.com/post/43624347354/matplotlib-trendline
        z = np.polyfit(norm, data, 1)
        p = np.poly1d(z)
        ax2.plot(norm, p(norm), color='r')
        ax2.set(**{'title': 'Normal Q-Q',
                   'xlabel': 'Standardized Residuals',
                   'ylabel': 'Theoretical Quantiles'})

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set(**{'title': 'Residual Order',
                   'xlabel': 'Observation',
                   'ylabel': 'Residuals'})
        ax3.plot(model_residuals, linestyle='--', alpha=0.5)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(model_leverage, model_cooks, color='r')
        ax4.scatter(x=model_leverage, y=model_norm_residuals, s=8, alpha=0.5)
        ax4.set(**{'title': '(DO NOT USE, NEEDS UPDATING: Residuals vs Leverage',
                   'xlabel': 'Leverage',
                   'ylabel': 'Standardized Residuals'})
        fig.subplots_adjust(hspace=.3)
        return fig

    @property
    def summary_stats(self) -> dict:
        """
        :return: RSE, adjusted r-squared, model p-value, etc.
        """
        rse = np.sqrt(np.sum(np.square(self._model_object.resid)) / self._model_object.df_resid)
        residuals = pd.Series(self._model_object.resid)

        return {'residual standard error (RSE)': rse,
                'adjusted r-squared': self._model_object.rsquared_adj,
                'model p-value': self._model_object.f_pvalue,
                # dividing Residual Standard Error ('average size' of errors) by the standard deviation of our
                # response variable, we can (sort of) get an idea of how large our errors are (are the average
                # errors less than the standard deviation of the response response variable?)
                'Ratio RSE to Target STD': rse / self._training_target_std,
                # this shifts the residuals by 1, so that we can take the correlation to see if previous
                # values are correlated with next/lagging values (i.e. e(i) provides no information about
                # e(i+1), where e is a residual), because the "standard errors that are computed for the
                # estimated regression coefficients or the fitted values are based on the assumption of
                # uncorrelated error terms (ISLR pg 93-94).
                'Residual Correlations':
                    pd.DataFrame({'1': residuals, '2': residuals.shift(1)}).dropna().corr().iloc[0, 1]}

    @property
    def results_summary(self) -> pd.DataFrame:
        """
        :return: a traditional view of the regression resampled_stats (feature coefficient estimates, p-values,
            etc.)
        """
        summary = pd.DataFrame(OrderedDict(zip(['(intercept)'] + self.feature_names,
                                               self._model_object.params.values.tolist())),
                               index=[0]).T
        summary.columns = ['Estimate']
        summary['Std. Error'] = self._model_object.bse.values  # round to 6
        summary['Pr(>|t|)'] = self._model_object.pvalues.values.round(5)  # round to 4
        summary['Sig'] = [self._get_significance_code(p_value=p_value) for p_value in summary['Pr(>|t|)']]
        return summary

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase=None) -> object:

        assert hyper_params is None  # no hyper-params for regression

        # Regression can't handle missing values
        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        if any(np.isnan(data_y)):
            raise MissingValueError()

        self._training_target_std = np.std(data_y)

        model_object = sm.OLS(data_y, sm.add_constant(data_x)).fit()
        return model_object

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return np.asarray(model_object.predict(sm.add_constant(data_x)))
