from oolearning.exploratory.ExploreDatasetBase import ExploreDatasetBase


class ExploreClassificationDataset(ExploreDatasetBase):
    # create method that compares the target variable against the feature of interest
    # for categorical feature, it should return a side by side bar chart
    # for a numeric feature, side-by-side box plot
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
