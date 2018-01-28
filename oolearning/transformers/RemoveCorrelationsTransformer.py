import pandas as pd
import numpy as np

from oolearning.transformers.TransformerBase import TransformerBase


class RemoveCorrelationsTransformer(TransformerBase):
    def __init__(self, max_correlation_threshold=0.95):
        super().__init__()
        self._max_correlation_threshold = max_correlation_threshold

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        # This method is described in APM on pg 47 as the following steps:
        #   - calculate the correlation matrix of features
        #   - determine the two features associated with the largest absolute pairwise correlation
        #     (call them features `A` and `B`)
        #   - Determine the average correlation between `A` and the other variables.
        #     - Do the same for `B`
        #   - If `A` has a larger average correlation, remove it; otherwise, remove feature `B`
        #   - Repeat until no absolute correlations are above the threshold (``r correlation_threshold``)

        columns_to_remove = list()

        while True:
            # noinspection PyUnresolvedReferences
            # `corr()` automatically excludes categorical features
            correlation_matrix = data_x.drop(columns=columns_to_remove, axis=1).corr()

            features = correlation_matrix.columns.values
            correlation_matrix = np.abs(correlation_matrix.as_matrix())
            np.fill_diagonal(correlation_matrix, np.NaN)

            # correlation_matrix.unique()
            # sorted(np.abs(np.unique(correlation_matrix)), reverse=True)

            highest_abs_pairwise_correlation = np.nanmax(correlation_matrix)

            if highest_abs_pairwise_correlation > self._max_correlation_threshold:
                # `where()` will always be 2 instances for correlation matrices, grab the first
                indexes = np.where(correlation_matrix == highest_abs_pairwise_correlation)[0]

                mean_a_correlation = np.nanmean(correlation_matrix[indexes[0], ])
                mean_b_correlation = np.nanmean(correlation_matrix[indexes[1], ])

                columns_to_remove.append(features[indexes[0]]
                                         if mean_a_correlation > mean_b_correlation
                                         else features[indexes[1]])
            else:
                break

        return {'columns_to_remove': columns_to_remove}

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return data_x.drop(columns=state['columns_to_remove'], axis=1)
