from typing import List

from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.hyper_params.HyperParamsGrid import HyperParamsGrid
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase


class ModelInfo:
    def __init__(self,
                 description: str,
                 model_wrapper: ModelWrapperBase,
                 transformations: List[TransformerBase]=None,
                 hyper_params: HyperParamsBase=None,
                 hyper_params_grid: HyperParamsGrid=None):
        self._description = description
        self._model_wrapper = model_wrapper
        self._transformations = transformations
        self._hyper_params = hyper_params
        self._hyper_params_grid = hyper_params_grid

    @property
    def description(self):
        return self._description

    @property
    def model_wrapper(self):
        return self._model_wrapper

    @property
    def transformations(self):
        return self._transformations

    @property
    def hyper_params(self):
        return self._hyper_params

    @property
    def hyper_params_grid(self):
        return self._hyper_params_grid
