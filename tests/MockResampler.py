import os
from typing import List

import dill as pickle
import numpy as np
import pandas as pd

from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.hyper_params.HyperParamsBase import HyperParamsBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase


class MockResampler(ResamplerBase):
    """
    This object mocks the tune_results from a previously ran RandomForest Tuner/Resampler
    """
    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase],
                 evaluators: List[EvaluatorBase]):
        super().__init__(model=model, model_transformations=model_transformations, evaluators=evaluators)
        file = os.path.join(os.getcwd(), 'tests/data/test_ModelTuner_classification_mock.pkl')
        with open(file, 'rb') as saved_object:
            self._tune_results = pickle.load(saved_object)

    # noinspection PyProtectedMember
    def _resample(self, data_x: pd.DataFrame, data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:

        # load actual data from a RandomForest Tuner/Resampler (test_ModelTuner_RandomForest_classification)
        # so that we can build up the necessary ResamplerResults object based on the saved data.

        lookup_df = self._tune_results._tune_results_objects  # get the actual tune_results table
        params_dict = hyper_params.params_dict  # get the dictionary associated with the current hyper-params

        # find the result index based off of the current hyper-params
        mock_matches = (lookup_df['criterion'] == params_dict['criterion']) & \
                       (lookup_df['max_features'] == params_dict['max_features']) & \
                       (lookup_df['n_estimators'] == params_dict['n_estimators']) & \
                       (lookup_df['min_samples_leaf'] == params_dict['min_samples_leaf'])

        mock_results = lookup_df[mock_matches]  # get the row that has the tune_results
        assert len(mock_results) == 1  # should only have one tune_results for a given set of hyper-parameters

        return mock_results.iloc[0].resampler_object
