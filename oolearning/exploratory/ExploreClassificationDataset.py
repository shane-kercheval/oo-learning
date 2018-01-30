import pandas as pd

from oolearning.exploratory.ExploreDatasetBase import ExploreDatasetBase


class ExploreClassificationDataset(ExploreDatasetBase):
    def __init__(self, dataset: pd.DataFrame, target_variable: str):
        super().__init__(dataset=dataset, target_variable=target_variable)
        if self._is_target_numeric:
            raise ValueError('the target variable cannot be numeric to use ExploreClassificationDataset')

    def compare_against_target(self, feature):
        """
        TODO: Document
        """
        assert feature != self._target_variable

        if feature in self._numeric_features:
            return self._dataset[[feature, self._target_variable]].boxplot(by=self._target_variable)
        else:
            return self._dataset.groupby([feature, self._target_variable]).size().unstack().\
                plot.bar(rot=10, title='{0} vs. target (`{1}`)'.format(feature, self._target_variable))
