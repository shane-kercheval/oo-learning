import pandas as pd
from sklearn import preprocessing

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


class NormalizationVectorSpaceTransformer(TransformerBase):
    """
        "Normalizes" the numeric features i.e. converts columns/features to have values ranging from 0-1.
    """
    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        return dict()  # normalization() doesn't fit

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        numeric_features, categoric_features = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,  # noqa
                                                                                     target_variable=None)

        data_x_numeric = pd.DataFrame(data=preprocessing.normalize(data_x[numeric_features]),
                                      columns=numeric_features,
                                      index=data_x.index)
        return pd.concat([data_x_numeric, data_x[categoric_features]], axis=1)
