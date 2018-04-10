from typing import List

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase


class ModelInfo:
    def __init__(self,
                 description: str,
                 model: ModelWrapperBase,
                 transformations: List[TransformerBase]=None,
                 hyper_params: HyperParamsBase=None,
                 hyper_params_grid: HyperParamsGrid=None,
                 converter: ContinuousToClassConverterBase=None):
        self._description = description
        self._model = model
        self._transformations = transformations
        self._hyper_params = hyper_params
        self._hyper_params_grid = hyper_params_grid
        self._converter = converter

    @property
    def description(self):
        return self._description

    @property
    def model(self):
        return self._model

    @property
    def transformations(self):
        return self._transformations

    @property
    def hyper_params(self):
        return self._hyper_params

    @property
    def hyper_params_grid(self):
        return self._hyper_params_grid

    @property
    def converter(self):
        return self._converter
