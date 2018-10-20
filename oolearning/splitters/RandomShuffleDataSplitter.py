from sklearn.model_selection import train_test_split

from math import ceil
from typing import Tuple

import numpy as np

from oolearning.splitters.DataSplitterBase import DataSplitterBase


class RandomShuffleDataSplitter(DataSplitterBase):
    def split(self, target_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param target_values: the target values of the full dataset you want to split
        :return: randomized/shuffled splits into training indexes (index 0 of tuple) and holdout indexes
            (index 1 of tuple); indexes are numeric locations (e.g. for index pandas DataFrame use `.iloc` not
            `.loc`)
        """
        indexes = np.arange(0, len(target_values))
        np.random.seed(self._seed)
        np.random.shuffle(indexes)
        cutoff = ceil(len(indexes) * (1 - self._holdout_ratio))
        return indexes[:cutoff], indexes[cutoff:]
