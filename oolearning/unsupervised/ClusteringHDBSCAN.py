from typing import Union

import numpy as np
import pandas as pd
import hdbscan

from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelExceptions import MissingValueError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class ClusteringHDBSCANHP(HyperParamsBase):
    """
    """
    def __init__(self, min_cluster_size: int=5, min_samples: Union[int, None]=None):
        super().__init__()
        self._params_dict = dict(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
                                )


class ClusteringHDBSCAN(ModelWrapperBase):

    def __init__(self, num_jobs: int=1):
        super().__init__()
        self._num_jobs = num_jobs

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y=None,
               hyper_params: HyperParamsBase = None) -> object:
        assert data_y is None

        # noinspection SpellCheckingInspection
        return "hdbscan.HDBSCAN doesn't have a `predict` function, only `fit_predict`, which I call in my `predict`"  # noqa

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> np.ndarray:
        data = data_x.copy()
        if data.isnull().sum().sum() > 0:
            raise MissingValueError()

        param_dict = self._hyper_params.params_dict
        model_object = hdbscan.HDBSCAN(
            min_cluster_size=param_dict['min_cluster_size'],
            min_samples=param_dict['min_samples'],
            core_dist_n_jobs=self._num_jobs
        )
        self._model_object = model_object
        return model_object.fit_predict(X=data)
