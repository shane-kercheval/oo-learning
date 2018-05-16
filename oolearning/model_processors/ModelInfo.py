from typing import List

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase


class ModelInfo:
    """
    Class that wraps/encapsulates model information, with the intent on handing the information to model
        processors, such as the ModelSearcher
    """
    def __init__(self,
                 model: ModelWrapperBase,
                 description: str=None,
                 transformations: List[TransformerBase]=None,
                 hyper_params: HyperParamsBase=None,
                 hyper_params_grid: HyperParamsGrid=None,
                 converter: ContinuousToClassConverterBase=None):
        """
        :param description: a *unique* description of the model (e.g. if forming a list of ModelInfo objects,
            then this description should be unique among the objects in the list)
        :param model:
        :param transformations:
        :param hyper_params: the hyper-params object
        :param hyper_params_grid: a HyperParamsGrid object, which specifies the different combinations of
            hyper-params to, for example, tune
        :param converter: A Converter object specifying how the predictions (e.g. DataFrame of probabilities
            for a classification problem) should be converted to classes.
        """
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

    def set_converter(self, converter: ContinuousToClassConverterBase=None):
        self._converter = converter
