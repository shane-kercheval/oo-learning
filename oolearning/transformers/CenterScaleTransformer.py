import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


class CenterScaleTransformer(TransformerBase):
    """
    Centers and Scales the numeric features (centers by subtracting the mean of the feature from each
        value within the feature; scales by dividing each value within the feature by the standard deviation
        of the feature.
    """
    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:

        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                    target_variable=None)

        averages = dict()
        standard_deviations = dict()
        # noinspection SpellCheckingInspection
        for feature in numeric_features:
            # APM pg 30
            # to center, the average of each feature is subtracted from all the values of that feature
            # save the average, so that the same average can be used to transform this and future datasets
            averages[feature] = data_x[feature].mean()
            # to scale, each value of each feature is divided by the standard deviation of that feature
            # save the st dev, so that the same st dev can be used to transform this and future datasets
            # `ddof=0` because:
            # https://stackoverflow.com/questions/44220290/sklearn-standardscaler-result-different-to-manual-result
            standard_deviations[feature] = data_x[feature].std(ddof=0)

        return dict(averages=averages, standard_deviations=standard_deviations)

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                    target_variable=None)
        averages = state['averages']
        standard_deviations = state['standard_deviations']

        for feature in numeric_features:
            # to center, the average of each feature is subtracted from all the values of that feature
            data_x[feature] = data_x[feature] - averages[feature]
            # to scale, each value of each feature is divided by the standard deviation of that feature
            data_x[feature] = data_x[feature] / standard_deviations[feature]

        return data_x
