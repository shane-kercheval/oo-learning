import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from oolearning.exploratory.ExploreDatasetBase import ExploreDatasetBase


class ExploreDataset(ExploreDatasetBase):

    def plot_against_target(self, feature: str):
        raise NotImplementedError('No target feature')

    def compare_numeric_summaries(self, other: pd.DataFrame) -> pd.DataFrame:
        """
        :param other: DataFrame (e.g. test set) to compare numeric summaries
        :return: if `x` is this .dataset's numeric_summary() and `y` is `other`'s numeric_summary() then this
            function returns `(y-x)/x`
        """
        # noinspection PyTypeChecker
        assert all(self.dataset.columns.values == other.columns.values)

        # needed because if the numeric_summary contains zero, we get NA; so we add a very small amount to
        # each value of each summary
        # should not affect the final outcome because we round to 5 digits; small possibility that adding
        # the constant results in a 0

        constant = 0.0000001
        self_numeric_summary = self.numeric_summary().apply(lambda x: x + constant)
        other_numeric_summary = ExploreDataset(other).numeric_summary().apply(lambda x: x + constant)
        diff = (other_numeric_summary - self_numeric_summary) / self_numeric_summary
        # need to drop these columns because they will be different simply because of the size of the dataset
        # which is already captured in `count` column

        return diff.drop(columns=['nulls', 'num_zeros']).round(5)

    def compare_categoric_summaries(self, other: pd.DataFrame) -> pd.DataFrame:
        """
        :param other: DataFrame (e.g. test set) to compare categoric summaries
        :return: if `x` is this .dataset's numeric_summary() and `y` is `other`'s numeric_summary() then this
            function returns `(y-x)/x`

            `self_diff_other` column represents all the categories that are in `self` but not in `other`
            `other_diff_self` column represents all the categories that are in `other` but not in `self`

            x_diff_y can be thought of set(x).difference(set(y))
        """
        # noinspection PyTypeChecker
        assert all(self.dataset.columns.values == other.columns.values)

        self_categoric_summary = self.categoric_summary()
        other_categoric_summary = ExploreDataset(other).categoric_summary()

        self_top_categories = self_categoric_summary.top
        other_top_categories = other_categoric_summary.top

        self_unique_categories = {x: set(self.dataset[x].values) for x in self.categoric_features}
        other_unique_categories = {x: set(other[x].values) for x in self.categoric_features}

        self_difference_other = [self_unique_categories[x].difference(other_unique_categories[x])
                                 for x in self.categoric_features]
        other_difference_self = [other_unique_categories[x].difference(self_unique_categories[x])
                                 for x in self.categoric_features]

        # needed because if the numeric_summary contains zero, we get NA; so we add a very small amount to
        # each value of each summary
        # should not affect the final outcome because we round to 5 digits; small possibility that adding
        # the constant results in a 0

        constant = 0.0000001
        columns_to_drop = ['nulls', 'top', 'unique']
        self_categoric_summary = self_categoric_summary.drop(columns=columns_to_drop).\
            apply(lambda x: x + constant)
        other_categoric_summary = other_categoric_summary.drop(columns=columns_to_drop).\
            apply(lambda x: x + constant)
        diff = (other_categoric_summary - self_categoric_summary) / self_categoric_summary
        diff = diff.round(5)

        diff['self_top_categories'] = self_top_categories
        diff['other_top_categories'] = other_top_categories
        diff['self_diff_other'] = pd.Series([str(x) if len(x) > 0 else '' for x in self_difference_other],
                                            index=self.categoric_features)
        diff['other_diff_self'] = pd.Series([str(x) if len(x) > 0 else '' for x in other_difference_self],
                                            index=self.categoric_features)

        return diff

    def compare_numeric_summaries_heatmap(self,
                                          other: pd.DataFrame,
                                          annotation_font_size: int = 8,
                                          y_axis_rotation: int = 0,
                                          axis_font_size: int = 10,
                                          plot_size: tuple = (12, 8),
                                          ):
        summary_diff = self.compare_numeric_summaries(other=other)
        sns.heatmap(summary_diff,
                    annot=summary_diff.apply(lambda x: x * 100).round(1).
                                       apply(lambda x: x.apply(lambda y: str(y) + '%')),
                    annot_kws={"size": annotation_font_size},
                    fmt='',  # fmt="f",
                    robust=True, cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                    cbar_kws={'label': 'Percent Change ([other - self] / self)'},
                    xticklabels=True,
                    yticklabels=True)
        plt.yticks(rotation=y_axis_rotation * -1,
                   va='center' if y_axis_rotation == 0 else 'bottom',
                   fontsize=axis_font_size)
        plt.xticks(rotation=30, ha='right', fontsize=axis_font_size)
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
        plt.tight_layout()

    def compare_numeric_boxplot(self, column: str, other: pd.DataFrame):
        x = [self.dataset[column].values,
             other[column].values]
        plt.boxplot(x, labels=['self', 'other'], whis='range')
        plt.title(column)
