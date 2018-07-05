import os
from typing import List

import dill as pickle
import numpy as np
import pandas as pd

from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.transformers.TransformerBase import TransformerBase
from tests.TestHelper import TestHelper


class MockResampler(ResamplerBase):
    """
    This object mocks the resampled_stats from a previously ran RandomForestClassifier Tuner/Resampler
    """
    def __init__(self,
                 model: ModelWrapperBase,
                 transformations: List[TransformerBase],
                 scores: List[ScoreBase]):
        super().__init__(model=model, transformations=transformations, scores=scores)
        # load actual data from a RandomForestClassifier Tuner/Resampler:
        # (test_ModelTuner_RandomForest_classification)
        # so that we can build up the necessary ResamplerResults object based on the saved data.
        file = os.path.join(os.getcwd(),
                            TestHelper.ensure_test_directory('data/test_ModelTuner_classification_mock.pkl'))
        with open(file, 'rb') as saved_object:
            self._tune_results = pickle.load(saved_object)

    # noinspection PyProtectedMember
    def _resample(self, data_x: pd.DataFrame, data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:

        lookup_df = self._tune_results._tune_results_objects  # get the actual resampled_stats table
        params_dict = hyper_params.params_dict  # get the dictionary associated with the current hyper-params

        # find the result index based off of the current hyper-params
        mock_matches = (lookup_df['criterion'] == params_dict['criterion']) & \
                       (lookup_df['max_features'] == params_dict['max_features']) & \
                       (lookup_df['n_estimators'] == params_dict['n_estimators']) & \
                       (lookup_df['min_samples_leaf'] == params_dict['min_samples_leaf'])

        mock_results = lookup_df[mock_matches]  # get the row that has the resampled_stats
        assert len(mock_results) == 1  # should only have one resampled_stats for a given set of hyper-parameters

        return mock_results.iloc[0].resampler_object
