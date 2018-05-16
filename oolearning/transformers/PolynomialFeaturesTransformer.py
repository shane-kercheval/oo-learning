import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from oolearning import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


class PolynomialFeaturesTransformer(TransformerBase):
    """
    Generates polynomial and interactions features for numeric features; retains categoric features.

    NOTE: must be used BEFORE DummyEncodeTransformer

    Utilizes `sklearn.PolynomialFeatures`:
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    """
    def __init__(self, degrees: int=2):
        """
        :param degrees: the degree of the polynomial feature
        """
        super().__init__()
        assert degrees > 1
        self._degrees = degrees

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        return {}

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        # include bias set to False (unlike underlying sklearn.PolynomialFeatures) because LinearRegressor
        # automatically adds the intercept column.
        poly_features = PolynomialFeatures(degree=self._degrees, include_bias=False)

        numeric_features, categorical_features = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,  # noqa
                                                                                       target_variable=None)
        poly_features.fit(X=data_x[numeric_features])
        new_features = poly_features.get_feature_names(numeric_features)
        transformed_x = pd.DataFrame(poly_features.transform(X=data_x[numeric_features]),
                                     columns=new_features,
                                     index=data_x.index)

        return pd.concat([transformed_x, data_x[categorical_features]], axis=1)
