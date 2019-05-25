from typing import Union, List

from oolearning.model_processors.SingleUseObject import Cloneable
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase


class CloneableFactory:
    def __init__(self, cloneable: Union[Cloneable, List[Cloneable]]):

        if isinstance(cloneable, list):
            for x in cloneable:
                assert x is None or isinstance(x, Cloneable)
        else:
            assert cloneable is None or isinstance(cloneable, Cloneable)

        self._cloneable = cloneable

    def get(self):
        if self._cloneable is None:
            return None

        if isinstance(self._cloneable, list):
            new_object = [x.clone() if x is not None else None for x in self._cloneable]
        else:
            new_object = self._cloneable.clone()

        return new_object


class ModelFactory(CloneableFactory):
    def __init__(self, model: ModelWrapperBase, hyper_params: HyperParamsBase = None):
        super().__init__(cloneable=[model, hyper_params])

        assert isinstance(model, ModelWrapperBase)
        assert hyper_params is None or isinstance(hyper_params, HyperParamsBase)

    def get_model(self):
        return self.get()[0]

    def get_hyper_params(self):
        return self.get()[1]


class TransformerFactory(CloneableFactory):
    def __init__(self, transformations: List[TransformerBase]):
        transformations = [x for x in transformations if x is not None] if transformations is not None else []
        super().__init__(cloneable=transformations)

        if transformations is not None:
            for x in transformations:
                assert x is None or isinstance(x, TransformerBase)

    def has_transformations(self):
        return self._cloneable is not None and len(self._cloneable) > 0

    def append_transformations(self, transformations: List[TransformerBase]):
        if transformations is not None:
            transformations = [x for x in transformations if x is not None]
            self._cloneable = self._cloneable + transformations
