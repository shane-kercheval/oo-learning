import pandas as pd

from oolearning.transformers.TransformerBase import TransformerBase


class RemoveNZPTransformer(TransformerBase):
    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        raise NotImplementedError()

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        pass
