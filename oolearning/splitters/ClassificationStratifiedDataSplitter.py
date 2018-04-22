import numpy as np
from oolearning.splitters.StratifiedDataSplitter import StratifiedDataSplitter


class ClassificationStratifiedDataSplitter(StratifiedDataSplitter):
    """
    Splits the data into training/holdout sets while maintaining the categorical proportions of the
    target variable
    """

    def labels_to_stratify(self, target_values: np.ndarray) -> np.ndarray:
        return target_values  # for classification, we are just going to use the target variables as is.
