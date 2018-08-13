import pandas as pd
import numpy as np

from oolearning.transformers.TransformerBase import TransformerBase


class RemoveCorrelationsTransformer(TransformerBase):
    """
    Removes numeric features that are correlated above a specified threshold. For each pair of features that
        are correlated above the specified threshold, the feature with the higher average correlation (among
        other features) is removed. The procedure is repeated until no correlation (absolute value) are above
        the threshold.
    """
    def __init__(self, max_correlation_threshold=0.95):
        """
        :param max_correlation_threshold: correlation threshold for algorithm
        """
        super().__init__()
        self._max_correlation_threshold = max_correlation_threshold

    def peak(self, data_x: pd.DataFrame):
        pass

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

        # noinspection PyUnresolvedReferences
        # `corr()` automatically excludes categorical features
        # we'll get the correlation outside the loop and remove features as we go because it is a very
        # expensive function call for large datasets
        correlation_matrix = data_x.corr()

        while True:

            local_correlation_matrix = correlation_matrix
            features = local_correlation_matrix.columns.values
            local_correlation_matrix = np.abs(local_correlation_matrix.values)
            np.fill_diagonal(local_correlation_matrix, np.NaN)

            # local_correlation_matrix.unique()
            # sorted(np.abs(np.unique(local_correlation_matrix)), reverse=True)

            highest_abs_pairwise_correlation = np.nanmax(local_correlation_matrix)

            if highest_abs_pairwise_correlation > self._max_correlation_threshold:
                # `where()` will always be 2 instances for correlation matrices, grab the first
                indexes = np.where(local_correlation_matrix == highest_abs_pairwise_correlation)[0]

                mean_a_correlation = np.nanmean(local_correlation_matrix[indexes[0], ])
                mean_b_correlation = np.nanmean(local_correlation_matrix[indexes[1], ])

                # A potential problem is that when we are e.g. resampling, there can be slight variations
                # depending on the scaling/etc.. and if, for example, the 'RemoveCorrelationsTransformer'
                # chooses (at "random") different features to remove, this messes with the functionality
                # that detects which features we should end with (e.g. a resampling training split doesn't
                # contain an uncommon value for a particular column, and is subsequently encoded
                # (e.g. one-hot) and then the training dataset does contain that value, and shit either breaks
                # or becomes inconsistent when predicting on two different transformed dataset
                # SO: we have to round (arbitrarily to 3) so that slight variations in correlations (e.g.
                # between the same two features when resampling) are consistent.
                if round(float(mean_a_correlation), 3) > round(float(mean_b_correlation), 3):
                    column_to_remove = features[indexes[0]]
                else:
                    column_to_remove = features[indexes[1]]
                columns_to_remove.append(column_to_remove)
                correlation_matrix.drop(index=column_to_remove, columns=column_to_remove, inplace=True)
            else:
                break

        return {'columns_to_remove': columns_to_remove}

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        # noinspection PyTypeChecker
        return data_x.drop(columns=state['columns_to_remove'])
