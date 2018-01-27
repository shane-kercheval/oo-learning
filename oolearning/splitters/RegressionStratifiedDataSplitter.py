import numpy as np
import pandas as pd

from oolearning.splitters.StratifiedDataSplitter import StratifiedDataSplitter


class RegressionStratifiedDataSplitter(StratifiedDataSplitter):
    """
    Splits the data into training/test groups while maintaining the distribution (think of a histogram) of the
    target variable
    """

    @staticmethod
    def bin_numeric_values(target_values, percentiles=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)):
        return pd.cut(target_values,
                      bins=[np.percentile(target_values, x) for x in percentiles],
                      right=True,
                      include_lowest=True).astype(str)

    def labels_to_stratify(self, target_values: np.ndarray) -> np.ndarray:
        return self.bin_numeric_values(target_values=target_values)
