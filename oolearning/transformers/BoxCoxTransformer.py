import pandas as pd

from scipy import stats

from oolearning.ModelSearcherHelpers import ModelSearcherHelpers
from oolearning.model_wrappers.ModelExceptions import NegativeValuesFoundError
from oolearning.transformers.TransformerBase import TransformerBase


# noinspection SpellCheckingInspection
class BoxCoxTransformer(TransformerBase):
    # TODO: document
    def __init__(self, predictors):
        """
        :param predictors: predictors i.e. column names to transform.
            The BoxCox transformation only works on non-negative data; if negative values are found in any of
            the `predictors`, a NegativeValuesFoundError will be raised when `fit()` is called.
        """
        super().__init__()
        self._predictors = predictors

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        numeric_predictors, _ = ModelSearcherHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                         target_variable=None)
        # make sure all of our predictors are numeric predictors
        assert all([predictor in numeric_predictors for predictor in self._predictors])

        lmbda_values = dict()
        for predictor in self._predictors:
            if any(data_x[predictor] < 0):
                raise NegativeValuesFoundError()
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
            # "If lmbda is None, find the lambda that maximizes the log-likelihood function and return it as
            # the second output argument."
            # so when fitting the data, we are going to ignore the transformation and get the lambda value
            # when transforming the data, we will used the saved lambda value as the input
            _, lmbda = stats.boxcox(x=data_x[predictor])
            lmbda_values[predictor] = lmbda

        return lmbda_values

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        for predictor in self._predictors:
            data_x[predictor] = stats.boxcox(x=data_x[predictor], lmbda=state[predictor])

        return data_x
