import numpy as np
import pandas as pd

from oolearning import HyperParamsBase, ResamplerResults
from oolearning.model_processors.ResamplerBase import ResamplerBase, ResamplerResults


class ClusteringResampler(ResamplerBase):

    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:
        assert data_y is None


        return ResamplerResults()