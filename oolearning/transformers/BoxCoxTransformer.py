import pandas as pd

from scipy import stats

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.model_wrappers.ModelExceptions import NegativeValuesFoundError
from oolearning.transformers.TransformerBase import TransformerBase


# noinspection SpellCheckingInspection
class BoxCoxTransformer(TransformerBase):
    """
    Applies the "Box-Cox" transformation to numeric data.

    uses https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
    """
    def __init__(self, features):
        """
        :param features: features i.e. column names to transform.
            The BoxCox transformation only works on non-negative data; if negative values are found in any of
            the `features`, a NegativeValuesFoundError will be raised when `fit()` is called.
        """
        super().__init__()
        self._features = features

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                    target_variable=None)
        # make sure all of our features are numeric features
        assert all([feature in numeric_features for feature in self._features])

        lmbda_values = dict()
        for feature in self._features:
            if any(data_x[feature] < 0):
                raise NegativeValuesFoundError()
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
            # "If lmbda is None, find the lambda that maximizes the log-likelihood function and return it as
            # the second output argument."
            # so when fitting the data, we are going to ignore the transformation and get the lambda value
            # when transforming the data, we will used the saved lambda value as the input
            _, lmbda = stats.boxcox(x=data_x[feature])
            lmbda_values[feature] = lmbda

        return lmbda_values

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        for feature in self._features:
            data_x[feature] = stats.boxcox(x=data_x[feature], lmbda=state[feature])

        return data_x
