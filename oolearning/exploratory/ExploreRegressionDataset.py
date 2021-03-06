import pandas as pd
import matplotlib.pyplot as plt

from oolearning.exploratory.ExploreDataset import ExploreDataset


class ExploreRegressionDataset(ExploreDataset):
    """
    ExploreRegressionDataset gives convenience while exploring a new dataset (with a numeric target
        variable) by providing common functionality frequently needed during standard exploration.


    WARNING: The underlying dataset should be changed from these class methods (i.e. subclass), rather
        than changing directly, since this class caches information about the dataset. If changes are made,
        the user can call `._update_cache()` manually.    
    """
    def __init__(self, dataset: pd.DataFrame, target_variable: str):
        super().__init__(dataset=dataset, target_variable=target_variable)
        if self.is_target_numeric is False:
            raise ValueError('the target variable must be numeric to use ExploreRegressionDataset')

    def plot_against_target(self, feature):
        """
        Shows a plot of the specific `feature` against, or compared with, the target variable.

        :param feature: feature to visualize and compare against the target
        """
        assert feature != self._target_variable

        if feature in self.numeric_features:
            self._dataset[[feature, self._target_variable]].plot.scatter(x=feature,
                                                                         y=self._target_variable,
                                                                         alpha=0.1,
                                                                         title='{0} vs. target (`{1}`)'.
                                                                         format(feature,
                                                                                self._target_variable))
        else:
            title = '{0} vs. target (`{1}`)'.format(feature, self.target_variable)
            self._dataset[[feature, self._target_variable]].boxplot(by=feature)
            plt.ylabel(self.target_variable)
            plt.title(title)
            plt.suptitle("")
        plt.tight_layout()

    def compare_target_boxplot(self, other: pd.DataFrame):
        self.compare_numeric_boxplot(column=self._target_variable, other=other)
        plt.tight_layout()
