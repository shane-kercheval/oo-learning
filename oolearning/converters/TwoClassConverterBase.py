from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd


class TwoClassConverterBase(metaclass=ABCMeta):
    @abstractmethod
    def convert(self,
                predicted_probabilities: pd.DataFrame,
                positive_class: object) -> np.ndarray:
        """
        Converts the `predicted_probabilities` into classes
        :param predicted_probabilities: `pd.DataFrame` that contains 2 columns, for each class. The column
            names must match the classes
        :param positive_class: the class that is considered the `positive` event
        :return: converted classes.
        """
        pass
