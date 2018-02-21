from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd


class ContinuousToClassConverterBase(metaclass=ABCMeta):
    @abstractmethod
    def convert(self, values: pd.DataFrame, **kwargs) -> np.ndarray:
        pass
