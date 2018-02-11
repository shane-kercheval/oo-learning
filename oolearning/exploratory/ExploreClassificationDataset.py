import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from oolearning.exploratory.ExploreDatasetBase import ExploreDatasetBase


class ExploreClassificationDataset(ExploreDatasetBase):
    def __init__(self, dataset: pd.DataFrame, target_variable: str):
        super().__init__(dataset=dataset, target_variable=target_variable)
        if self._is_target_numeric:
            raise ValueError('the target variable cannot be numeric to use ExploreClassificationDataset')

    def plot_histogram_against_target(self, numeric_feature):
        # percentiles = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
        subset = pd.DataFrame(pd.cut(self._dataset[numeric_feature],
                                     bins=20,
                                     # [np.percentile(self._dataset[feature], x) for x in percentiles],
                                     right=True,
                                     include_lowest=True))
        subset[self._target_variable] = self._dataset[self._target_variable]
        return self._dodged_barchart(dataset=subset,
                                     feature=numeric_feature,
                                     target_variable=self._target_variable)

    def plot_against_target(self, feature):
        """
        TODO: Document
        """
        assert feature != self._target_variable

        if feature in self._numeric_features:
            return self._dataset[[feature, self._target_variable]].boxplot(by=self._target_variable)
        else:
            return self._dodged_barchart(dataset=self._dataset,
                                         feature=feature,
                                         target_variable=self._target_variable)

    @staticmethod
    def _dodged_barchart(dataset: pd.DataFrame, feature, target_variable):
        """
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

        ax_totals = ax.bar(group_locations + (width / num_of_unique_classes),
                           totals, width * num_of_unique_classes, color='black', alpha=0.15)

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
        ax.set_xticklabels(labels=labels, rotation=20)

        ax.legend([ax[0] for ax in ax_list] + [ax_totals[0]], [x for x in unique_classes] + ['Total'],
                  title=target_variable)
        return ax
