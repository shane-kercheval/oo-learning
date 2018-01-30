import pandas as pd

from oolearning import ExploreDatasetBase


class ExploreRegressionDataset(ExploreDatasetBase):
    def __init__(self, dataset: pd.DataFrame, target_variable: str):
        super().__init__(dataset=dataset, target_variable=target_variable)
        if self._is_target_numeric is False:
            raise ValueError('the target variable must be numeric to use ExploreRegressionDataset')

    def compare_against_target(self, feature):
        """
        TODO: Document
        """
        assert feature != self._target_variable

        if feature in self._numeric_features:
            return self._dataset[[feature, self._target_variable]].plot.\
                scatter(x=feature, y=self._target_variable, alpha=0.1,
                        title='{0} vs. target (`{1}`)'.format(feature, self._target_variable))
        else:
            return self._dataset[[feature, self._target_variable]].boxplot(by=feature)
