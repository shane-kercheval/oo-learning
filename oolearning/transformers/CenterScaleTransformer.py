import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


# TODO: document
class CenterScaleTransformer(TransformerBase):
    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:

        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                    target_variable=None)
        # APM pg 30
        # to center, the average of each feature is subtracted from all the values of that feature
        # save the average, so that the same average can be used to transform this and future datasets
        averages = dict()
        for feature in numeric_features:
            averages[feature] = data_x[feature].mean()
        # to scale, each value of each feature is divided by the standard deviation of that feature
        # save the st dev, so that the same st dev can be used to transform this and future datasets
        standard_deviations = dict()
        for feature in numeric_features:
            standard_deviations[feature] = data_x[feature].std()

        return dict(averages=averages, standard_deviations=standard_deviations)

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                    target_variable=None)
        averages = state['averages']
        standard_deviations = state['standard_deviations']

        # to center, the average of each feature is subtracted from all the values of that feature
        for feature in numeric_features:
            data_x[feature] = data_x[feature] - averages[feature]

        # to scale, each value of each feature is divided by the standard deviation of that feature
        for feature in numeric_features:
            data_x[feature] = data_x[feature] / standard_deviations[feature]

        return data_x
