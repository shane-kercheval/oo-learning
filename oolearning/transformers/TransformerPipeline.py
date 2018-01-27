import pandas as pd
from typing import List

from oolearning.transformers.TransformerBase import TransformerBase


class TransformerPipeline:
    """
    Transformer is like a sklearn Transformer/Pipeline, but doesn't lose column names.
    And there is extra logic to ensure order is done correctly, etc..
    There is a need to have a Transformer on the entire data, and then model specific.

    Recommended Order:
        # order APL pg 55?
        # transformation
        # normalization (center then scale)
        # imputation,
        # feature extraction
        # FE - remove NZV
        # FE - remove correlation
        # FE - pca
        # custom_post_function
        # ResolveOutliers (? )
        # spatial sign (even though this is a transformation, it needs to happen after normalization and any
        # feature extraction (APL pg 34)
    """

    def __init__(self, transformations: List[TransformerBase]):
        assert transformations is None or isinstance(transformations, list)
        if transformations is not None:
            assert all([isinstance(x, TransformerBase) for x in transformations])
        self._transformations = transformations
        self._has_fitted_values = None

    @property
    def transformations(self) -> List[TransformerBase]:
        return self._transformations

    def append_pipeline(self, transformation_pipeline: 'TransformerPipeline'):
        """
        adds all the transformations passed in to the current object.
        appends the addition transformations to the end of the current pipeline list, and maintains order
        :param transformation_pipeline: a TransformerPipeline object
        :return:
        """
        assert isinstance(transformation_pipeline, TransformerPipeline)
        if self._transformations is None:
            if transformation_pipeline.transformations is None:  # both transformation lists are None
                return
            else:  # if current object's transformations is None and we have shit to add, must create the list
                self._transformations = []

        if transformation_pipeline.transformations is not None:
            self._transformations.extend(transformation_pipeline.transformations)

    def fit(self):
        """
        It's not possible to `fit` all the individual transformers alone, otherwise subsequent transformers
        would be fitting on untransformed data and would fit the incorrect values.
        """
        raise NotImplementedError()

    def fit_transform(self, data_x: pd.DataFrame) -> pd.DataFrame:
        """
        cycles through each Transformation in the pipeline, fits (i.e. saves the transformation information
        for subsequent transforms) and then transforms the data, and passes the transformed data to the next
        Transformation object.
        :param data_x: DataFrame to fit/transform
        :return: transformed DataFrame
        """
        assert self._has_fitted_values is None  # i.e. we should only call once (e.g. on training, not test)
        data_x = data_x.copy()  # creates a copy so regardless if we execute the for, we still return a copy
        if self._transformations is not None:
            for transformation in self._transformations:
                data_x = transformation.fit_transform(data_x=data_x)

        self._has_fitted_values = True
        return data_x

    def transform(self, data_x: pd.DataFrame) -> pd.DataFrame:
        """
         cycles through each Transformation in the pipeline, and transforms the data, passing the transformed
         data to the next Transformation object
        :param data_x: DataFrame to transform
        :return: transformed DataFrame
        """
        data_x = data_x.copy()  # creates a copy so regardless if we execute the for, we still return a copy
        assert self._has_fitted_values is True  # i.e. assert that we have called fit_transform
        if self._transformations is not None:
            for transformation in self._transformations:
                data_x = transformation.transform(data_x=data_x)

        return data_x

    @classmethod
    def get_expected_columns(cls, transformations: List[TransformerBase], data: pd.DataFrame) -> List[str]:
        trans_copy = [x.clone() for x in transformations]
        pipeline = cls(transformations=trans_copy)
        transformed_data = pipeline.fit_transform(data_x=data)
        return transformed_data.columns.values.tolist()
