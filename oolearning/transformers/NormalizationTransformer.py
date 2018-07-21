import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


class NormalizationTransformer(TransformerBase):
    """
        "Normalizes" the numeric features i.e. converts columns/features to have values ranging from 0-1.
    """
    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:

        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                    target_variable=None)
        # Hands On Machine Learning pg 67
        # Values are shifted and rescaled so that they end up ranging from 0 to 1.
        # We do this by subtracting the min value and dividing by the max minus the min.
        minimums = dict()
        maximums = dict()
        for feature in numeric_features:
            minimums[feature] = data_x[feature].min()
            maximums[feature] = data_x[feature].max()

        return dict(minimums=minimums, maximums=maximums)

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                    target_variable=None)
        minimums = state['minimums']
        maximums = state['maximums']

        # normalize by subtracting the min value and dividing by the max minus the min.
        for feature in numeric_features:
            data_x[feature] = (data_x[feature] - minimums[feature]) / (maximums[feature] - minimums[feature])

        return data_x
