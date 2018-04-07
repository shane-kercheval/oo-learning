from typing import Union

import pandas as pd
import numpy as np

from oolearning import ModelWrapperBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase


class ModelStacker(ModelWrapperBase):
    def __init__(self):
        super().__init__()

    @property
    def feature_importance(self):
         raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        pass

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        pass



# TODO: we could either use the probabilities (positive class for two-class) or labels
# not sure we can use probabilities for multi-class?
