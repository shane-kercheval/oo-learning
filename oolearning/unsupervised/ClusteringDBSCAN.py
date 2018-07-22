import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

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

    def __init__(self, num_jobs: int=1, seed: int=42):
        super().__init__()
        self._num_jobs = num_jobs
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

        param_dict = self._hyper_params.params_dict
        model_object = DBSCAN(eps=param_dict['epsilon'],
                              min_samples=param_dict['min_samples'],
                              n_jobs=self._num_jobs)
        self._model_object = model_object
        return model_object.fit_predict(X=data)
