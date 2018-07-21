from enum import unique, Enum
from typing import Union

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


@unique
class ClusteringHierarchicalLinkage(Enum):
    WARD = 'ward'
    COMPLETE = 'complete'
    AVERAGE = 'average'


@unique
class ClusteringHierarchicalAffinity(Enum):
    EUCLIDEAN = 'euclidean'
    L1 = 'l1'
    L2 = 'l2'
    MANHATTAN = 'manhattan'
    COSINE = 'cosine'
    PRECOMPUTED = 'precomputed'


class ClusteringHierarchicalHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more information
        on tuning parameters
    """
    def __init__(self,
                 num_clusters: int=2,
                 linkage: ClusteringHierarchicalLinkage=ClusteringHierarchicalLinkage.WARD,
                 affinity: Union[ClusteringHierarchicalAffinity,
                                 callable]=ClusteringHierarchicalAffinity.EUCLIDEAN):
        super().__init__()
        if linkage == ClusteringHierarchicalLinkage.WARD:
            assert affinity == ClusteringHierarchicalAffinity.EUCLIDEAN

        self._params_dict = dict(
            num_clusters=num_clusters,
            linkage=linkage.value,
            affinity=affinity.value,
        )


class ClusteringHierarchical(ModelWrapperBase):

    def __init__(self, shuffle_data: bool=True, seed: int=42):
        super().__init__()
        self._shuffle_data = shuffle_data
        self._seed = seed

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y=None,
               hyper_params: HyperParamsBase = None) -> object:
        assert data_y is None

        return "sklearn doesn't have a `predict` function, only `fit_predict`, which I call in my `predict`"

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> np.ndarray:
        data = data_x.copy()
        if data.isnull().sum().sum() > 0:
            raise MissingValueError()

        # in ClusteringKmeans, because sklearn has a `predict()`, shuffling the data in `train()` has no
        # affect on predict because the data in `train()` is shuffled, but the data in `predict()` is not
        # so what is returned is the same order has what is passed in
        # here, we are shuffling in predict, so we have to un-shuffle and return the clusters in the original
        # order
        original_indexes = data_x.index.values

        if self._shuffle_data:
            # "randomly" shuffle data
            # noinspection PyTypeChecker
            indexes = np.arange(0, len(data))
            np.random.seed(self._seed)
            np.random.shuffle(indexes)
            data = data.iloc[indexes]

        param_dict = self._hyper_params.params_dict
        model_object = AgglomerativeClustering(
            n_clusters=param_dict['num_clusters'],
            linkage=param_dict['linkage'],
            affinity=param_dict['affinity'],
        )
        predictions = model_object.fit_predict(X=data)

        predictions_df = pd.DataFrame(data={'clusters': predictions}, index=data.index)
        return predictions_df.loc[original_indexes].clusters.values
