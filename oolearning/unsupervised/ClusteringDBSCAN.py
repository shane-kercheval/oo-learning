import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class ClusteringDBSCANHP(HyperParamsBase):
    """
    """
    def __init__(self,  epsilon: float=0.5, min_samples: int=5):
        super().__init__()
        self._params_dict = dict(
                                    epsilon=epsilon,
                                    min_samples=min_samples,
                                )


class ClusteringDBSCAN(ModelWrapperBase):

    def __init__(self, shuffle_data: bool=True, num_jobs: int=1, seed: int=42):
        super().__init__()
        self._shuffle_data = shuffle_data
        self._num_jobs = num_jobs
        self._seed = seed
        self._silhouette_score = None

    @property
    def silhouette_score(self):
        return self._silhouette_score

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
        model_object = DBSCAN(eps=param_dict['epsilon'],
                              min_samples=param_dict['min_samples'],
                              n_jobs=self._num_jobs)
        clusters = model_object.fit_predict(X=data)
        # create a dataframe so we can easily sort back to the original indexes, to return clusters in
        # the expected order
        predictions_df = pd.DataFrame(data={'clusters': clusters}, index=data.index)
        predictions_df = predictions_df.loc[original_indexes]
        if len(np.unique(clusters)) >= 2:  # silhouette_score requires >=2 clusters
            self._silhouette_score = silhouette_score(X=data_x, labels=predictions_df.clusters.values)
        else:
            self._silhouette_score = -1
        return predictions_df.clusters.values
