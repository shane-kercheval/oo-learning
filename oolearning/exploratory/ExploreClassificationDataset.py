import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from oolearning.exploratory.ExploreDatasetBase import ExploreDatasetBase


class ExploreClassificationDataset(ExploreDatasetBase):
    """
    ExploreClassificationDataset gives convenience while exploring a new dataset (with a categoric target
        variable) by providing common functionality frequently needed during standard exploration.

    
    WARNING: The underlying dataset should be changed from these class methods (i.e. subclass), rather
        than changing directly, since this class caches information about the dataset. If changes are made,
        the user can call `._update_cache()` manually.    
    """
    def __init__(self,
                 dataset: pd.DataFrame,
                 target_variable: str,
                 map_numeric_target: dict=None,
                 ordered: bool=False):
        """
        :param dataset: dataset to explore
        :param target_variable: the name of the target variable/column
        :param map_numeric_target: A dictionary mapping that describes how to convert a numeric target (e.g. 
            a column with `0`s & `1`s to e.g. a column of `yes`s & `no`s.).

            for example, the dictionary for the example above would be `{0: 'no', 1: 'yes'}`
        :param ordered: when `map_numeric_target` is not None, `ordered` is a flag indicating whether or not
            the provided dictionary (in `map_numeric_target`) should be treated in the DataFrame column 
            (pd.Categorical) as logically ordered.
        """
        super().__init__(dataset=dataset, target_variable=target_variable)
        if self._is_target_numeric and map_numeric_target is not None:
            self.set_as_categoric(feature=target_variable, mapping=map_numeric_target, ordered=ordered)

    @classmethod
    def from_csv(cls,
                 csv_file_path: str,
                 target_variable: str,
                 map_numeric_target: dict=None,
                 ordered: bool=False) -> 'ExploreDatasetBase':
        """
        Instantiates this class (via subclass) by first loading in a csv from `csv_file_path`.

        NOTE: this method sets non-numeric columns to `pd.Categorical` types.

        :param csv_file_path: path to the csv file
        :param target_variable: the name of the target variable/column
         :param map_numeric_target: same as `__init__()`
        :param ordered: same as `__init__()`
        :return: an instance of this class (i.e. subclass)
        """
        explore = super().from_csv(csv_file_path=csv_file_path, target_variable=target_variable)
        if explore._is_target_numeric:
            if map_numeric_target is None:
                raise ValueError('need to provide `map_numeric_target` for numeric targets')
            explore.set_as_categoric(feature=target_variable, mapping=map_numeric_target, ordered=ordered)

        return explore

    def plot_histogram_against_target(self, numeric_feature):
        """
        Shows a plot of the specific `numeric_feature` against, or compared with, the target variable, as a
            histogram.

        :param numeric_feature: feature to visualize and compare against the target
        """
        subset = pd.DataFrame(pd.cut(self._dataset[numeric_feature],
                                     bins=20,
                                     # [np.percentile(self._dataset[feature], x) for x in percentiles],
                                     right=True,
                                     include_lowest=True))
        subset[self._target_variable] = self._dataset[self._target_variable]
        self._dodged_barchart(dataset=subset,
                              feature=numeric_feature,
                              target_variable=self._target_variable)

    def plot_against_target(self, feature):
        """
        Shows a plot of the specific `feature` against, or compared with, the target variable.

        :param feature: feature to visualize and compare against the target
        """
        assert feature != self._target_variable

        if feature in self._numeric_features:
            self._dataset[[feature, self._target_variable]].boxplot(by=self._target_variable)
        else:
            self._dodged_barchart(dataset=self._dataset,
                                  feature=feature,
                                  target_variable=self._target_variable,
                                  plot_group_percentages=True)

    @staticmethod
    def _dodged_barchart(dataset: pd.DataFrame, feature, target_variable, plot_group_percentages=False):
        """
        Helper method that creates a dodged barchart.
        :param: dataset containing the feature column and target_variable
        :return: bar chart
        """
        grouped_data = dataset.groupby([feature, target_variable]).size()
        labels = [x for x in grouped_data.index.get_level_values(feature).unique()]
        number_of_feature_classes = len(labels)

        totals = [grouped_data[index].sum() for index in labels]
        # men_means = (20, 35, 30, 35, 27)
        # women_means = (25, 32, 34, 20, 25)

        group_locations = np.arange(number_of_feature_classes)  # the x locations for the groups
        # todo: will this width work with > 2 classes?
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()

        unique_classes = dataset[target_variable].unique()
        num_of_unique_classes = len(unique_classes)

        bar_midpoints = group_locations + (width / num_of_unique_classes)
        ax_totals = ax.bar(bar_midpoints,
                           totals, width * num_of_unique_classes, color='black', alpha=0.15)
        if plot_group_percentages:
            grouped_data_totals = dataset.groupby([feature]).size()
            grouped_data_percentages = grouped_data / grouped_data_totals
            total_bar_labels = ['{0}% | {1}%'.format(round(grouped_data_percentages.loc[x].iloc[0] * 100, 1),
                                                     round(grouped_data_percentages.loc[x].iloc[1] * 100, 1))
                                for x in labels]
            for i, v in enumerate(totals):
                # noinspection PyUnboundLocalVariable
                ax.text(bar_midpoints[i], v + 1, total_bar_labels[i], color='black', ha='center')

        # from matplotlib.pyplot import cm
        # colors = cm.rainbow(np.linspace(0, 1, 10)[::-1])
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        if num_of_unique_classes > len(colors):
            raise NotImplementedError(
                'Need to update implementation to use >' + str(len(colors)) + ' colors :(')

        ax_list = []
        for index in range(num_of_unique_classes):
            # the 'if' is because it's not guaranteed that every class of the target variable will be found
            # in each category of 'feature', especially if the category is very small in size.
            counts = [grouped_data[x, unique_classes[index]]
                      if unique_classes[index] in grouped_data.loc[x].index else 0
                      for x in labels]
            ax_list.append(
                ax.bar(group_locations + (width * index), tuple(counts), width, color=colors[index]))

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Count')
        ax.set_xlabel(feature)
        ax.set_title('{0} vs. target (`{1}`)'.format(feature, target_variable))
        ax.set_xticks(group_locations + width / num_of_unique_classes)
        ax.set_xticklabels(labels=labels, rotation=20, ha='right')

        ax.legend([ax[0] for ax in ax_list] + [ax_totals[0]], [x for x in unique_classes] + ['Total'],
                  title=target_variable)
        return ax
