from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd


class ContinuousToClassConverterBase(metaclass=ABCMeta):
    """
    A Converter converts a DataFrame containing predictions (with continuous values (e.g. probabilities)
        for each class, as columns) into an array of class predictions.


    e.g.

    Titanic data-set is trained on a model with the goal of predicting who Survived; classes are "lived" and
        "died".

    The output of `.predict()` is a DataFrame in the following format:

    ```
    died  | lived
    ====    =====
    0       1
    0.6     0.4
    0.1     0.9
    ...     ...
    ```

    And the Converter might convert that DataFrame to `lived, died, lived, ...`
    """

    @abstractmethod
    def convert(self, values: pd.DataFrame) -> np.ndarray:
        """
        :param values: the (DataFrame) output of a model's `.predict()`, which has column names as class names
        :return: an array of class predictions.
        """
        pass
