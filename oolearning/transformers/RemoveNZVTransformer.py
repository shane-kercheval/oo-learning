import pandas as pd
import numpy as np

from oolearning.transformers.TransformerBase import TransformerBase


# noinspection PyTypeChecker, SpellCheckingInspection
class RemoveNZVTransformer(TransformerBase):
    """
    Removes 'near zero variance' (NZV) numeric features; NZV defined as features that have a `standard
        deviation / absolute value of mean` ratio that is less than the specified value.
    """
    def __init__(self, stdev_to_mean_ratio: float=0.02):
        """
        :param stdev_to_mean_ratio: columns that have a `standard deviation / absolute value of mean` ratio
            that is less than `stdev_to_mean_ratio` will be removed.
        """
        super().__init__()
        self._stdev_to_mean_ratio = stdev_to_mean_ratio

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        means = data_x.mean()
        st_devs = data_x.std()
        assert all(means.index.values == st_devs.index.values)  # ensure same order

        mean_values = [x if x != 0 else 0.001 for x in data_x.mean().values]  # ensure no divide by zero
        columns_to_remove = list(means[st_devs.values / np.abs(mean_values) < self._stdev_to_mean_ratio].index.values)  # noqa

        return dict(columns_to_remove=columns_to_remove)

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        return data_x.drop(columns=state['columns_to_remove'])
