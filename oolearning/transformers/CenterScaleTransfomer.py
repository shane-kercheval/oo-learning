import pandas as pd

from ..ModelSearcherHelpers import ModelSearcherHelpers
from .TransformerBase import TransformerBase


# TODO: document
class CenterScaleTransformer(TransformerBase):
    def _fit_definition(self, data_x: pd.DataFrame) -> dict:

        numeric_predictors, _ = ModelSearcherHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                         target_variable=None)
        # APM pg 30
        # to center, the average of each predictor is subtracted from all the values of that predictor
        # save the average, so that the same average can be used to transform this and future datasets
        averages = dict()
        for predictor in numeric_predictors:
            averages[predictor] = data_x[predictor].mean()
        # to scale, each value of each predictor is divided by the standard deviation of that predictor
        # save the st dev, so that the same st dev can be used to transform this and future datasets
        standard_deviations = dict()
        for predictor in numeric_predictors:
            standard_deviations[predictor] = data_x[predictor].std()

        return dict(averages=averages, standard_deviations=standard_deviations)

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        numeric_predictors, _ = ModelSearcherHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                         target_variable=None)
        averages = state['averages']
        standard_deviations = state['standard_deviations']

        # to center, the average of each predictor is subtracted from all the values of that predictor
        for predictor in numeric_predictors:
            data_x[predictor] = data_x[predictor] - averages[predictor]

        # to scale, each value of each predictor is divided by the standard deviation of that predictor
        for predictor in numeric_predictors:
            data_x[predictor] = data_x[predictor] / standard_deviations[predictor]

        return data_x
