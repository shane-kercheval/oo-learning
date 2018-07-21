from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_wrappers.SklearnPredictMixin import SklearnPredictArrayMixin


class ClusteringKMeansHP(HyperParamsBase):
    """
    See http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html for more information
        on tuning parameters
    """
    def __init__(self,
                 num_clusters: int=8,
                 init_method: Union[str, np.ndarray]='k-means++',
                 num_different_seeds: int=10,
                 max_iterations: int=300,
                 precompute_distances: Union[str, bool]='auto',
                 algorithm: str='auto'):
        super().__init__()
        self._params_dict = dict(
                                    num_clusters=num_clusters,
                                    init_method=init_method,
                                    num_different_seeds=num_different_seeds,
                                    max_iterations=max_iterations,
                                    precompute_distances=precompute_distances,
                                    algorithm=algorithm,
                                )


class ClusteringKMeans(SklearnPredictArrayMixin, ModelWrapperBase):

    def __init__(self, evaluate_bss_tss: bool=False, shuffle_data: bool=True, num_jobs: int=1, seed: int=42):
        super().__init__()
        self._shuffle_data = shuffle_data
        self._num_jobs = num_jobs
        self._seed = seed
        self._score = None
        self._evaluate_bss_tss = evaluate_bss_tss
        self._bss = None
        self._tss = None
        self._wss = None

    @property
    def feature_importance(self):
        raise NotImplementedError()

    @property
    def score(self):
        """
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        :return: Opposite of the value of X on the K-means objective.
        """
        return self._score

    @property
    def bss(self):
        return None if self.tss is None or self.wss is None else self.tss - self.wss

    @property
    def tss(self):
        return self._tss

    @property
    def wss(self):
        return self._wss

    @property
    def bss_tss_ratio(self):
        return None if self.bss is None else self.bss / self.tss

    def _train(self,
               data_x: pd.DataFrame,
               data_y=None,
               hyper_params: HyperParamsBase = None) -> object:
        assert data_y is None

        if data_x.isnull().sum().sum() > 0:
            raise MissingValueError()

        if self._shuffle_data:
            # "randomly" shuffle data
            indexes = np.arange(0, len(data_x))
            np.random.seed(self._seed)
            np.random.shuffle(indexes)
            data_x = data_x.iloc[indexes]

        param_dict = hyper_params.params_dict
        model_object = KMeans(
            n_clusters=param_dict['num_clusters'],
            init=param_dict['init_method'],
            n_init=param_dict['num_different_seeds'],
            max_iter=param_dict['max_iterations'],
            precompute_distances=param_dict['precompute_distances'],
            algorithm=param_dict['algorithm'],
            n_jobs=self._num_jobs,
            random_state=self._seed)
        model_object.fit(X=data_x)

        self._score = model_object.score(X=data_x)

        if self._evaluate_bss_tss:
            # distances for each data-point to each cluster center
            distances = cdist(data_x, model_object.cluster_centers_, 'euclidean')
            min_distances = np.min(distances, axis=1)

            # Total with-in sum of square
            self._wss = sum(min_distances ** 2)  # ends of being absolute value of self._score
            self._tss = sum(pdist(data_x) ** 2) / data_x.shape[0]

        return model_object
