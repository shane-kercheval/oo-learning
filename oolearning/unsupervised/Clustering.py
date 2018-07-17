from enum import Enum, unique
from typing import List

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from oolearning.model_processors.ModelFitter import ModelFitter
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.CenterScaleTransformer import CenterScaleTransformer
from oolearning.unsupervised.ClusteringKmeans import ClusteringKMeans, ClusteringKMeansHP


@unique
class ClusteringHeatmapTransStrategy(Enum):
    PERCENTILES = 0,
    CENTER_SCALE = 1,


@unique
class ClusteringHeatmapValues(Enum):
    STRATEGY = 0,
    ACTUAL = 1,


@unique
class ClusteringHeatmapAggStrategy(Enum):
    MEAN = 0,
    MEDIAN = 1,


class Clustering:
    @staticmethod
    def cluster_heatmap(data: pd.DataFrame,
                        clusters: np.array,
                        trans_strategy: ClusteringHeatmapTransStrategy.CENTER_SCALE,
                        agg_strategy: ClusteringHeatmapAggStrategy.MEAN,
                        display_values: ClusteringHeatmapValues.ACTUAL):
        """
        :type agg_strategy: This strategy (ClusteringHeatmapAggStrategy) will determine how the values for
            each feature/cluster are aggregated.
            Specifically, each cell in the heatmap corresponds to, for example, the average (or median, etc.)
                value for that cluster/feature. The strategy determines if it is mean/median/etc.
        :param display_values:
            ClusteringHeatmapValues.STRATEGY displays the values according to the
                `ClusteringHeatmapTransStrategy` passed in.
            ClusteringHeatmapValues.ACTUAL displays the actual values according to the `agg_strategy`.
                NOTE: when set to ACTUAL, the colormap/legend will still be based on the
                ClusteringHeatmapTransStrategy, not on the values inside the cells.
        :param trans_strategy: The strategy determines how the heatmap is colored (and if display_values is
            set to ClusteringHeatmapValues.STRATEGY, it determines which values are displayed)

            ClusteringHeatmapTransStrategy.PERCENTILES converts each column/feature to percentiles before
                aggregating and displaying
            ClusteringHeatmapTransStrategy.CENTER_SCALE centers and scales each column/feature (i.e.
                transforms to corresponding z-score values before aggregating and displaying

        :param data: data from which the clusters were generated
            It is recommended you use the pre-transformed data (i.e. could contain missing values), otherwise
                mean/median/etc. is calculated on filled values (i.e. replacing NAs with 0) which will change
                the aggregated values.

                e.g. if original column was `(0, 1, 0, 1, NA, NA)` and transformed column was
                    `(0, 1, 0, 1, 0, 0)` then the original mean is `0.5` while the mean for the transformed
                    column is `0.333`
        :param clusters:
        :return:
        """
        if agg_strategy == ClusteringHeatmapAggStrategy.MEAN:
            agg_method = pd.DataFrame.mean
            agg_strategy_label = 'Mean'

        elif agg_strategy == ClusteringHeatmapAggStrategy.MEDIAN:
            agg_method = pd.DataFrame.median
            agg_strategy_label = 'Median'
        else:
            raise NotImplementedError()

        if trans_strategy == ClusteringHeatmapTransStrategy.CENTER_SCALE:
            transformed_data = CenterScaleTransformer().fit_transform(data)
            color_scale_min = -2
            color_scale_max = 2
            color_scale_title = "Cluster's {} Stan. Dev. (Z-Score) of Feature".format(agg_strategy_label)

        elif trans_strategy == ClusteringHeatmapTransStrategy.PERCENTILES:
            transformed_data = data.rank(pct=True)
            color_scale_min = 0
            color_scale_max = 1
            color_scale_title = "Cluster's {} Percentile Value for the Feature".format(agg_strategy_label)

        else:
            raise NotImplementedError()

        transformed_data['cluster'] = clusters

        cluster_size = transformed_data.groupby('cluster').size()
        cluster_size_lookup = cluster_size.values

        group_data = transformed_data.groupby('cluster').apply(agg_method).drop(columns='cluster')
        indexes_with_sizes = ['{1} ({0})'.format(index, size) for index, size in zip(group_data.index.values,
                                                                                   cluster_size_lookup)]
        group_data.index = indexes_with_sizes

        if display_values == ClusteringHeatmapValues.ACTUAL:
            temp = data
            temp['cluster'] = clusters
            temp.groupby('cluster').apply(pd.DataFrame.median)
            values = temp.groupby('cluster').apply(agg_method).drop(columns='cluster')
            values.index = indexes_with_sizes

        elif display_values == ClusteringHeatmapValues.STRATEGY:
            values = group_data

        else:
            raise NotImplementedError()

        values = values.round(2)
        for column in values.columns.values:
            values[column] = values[column].astype(str)

        sns.heatmap(group_data.iloc[np.argsort(cluster_size_lookup)].transpose(),
                    annot=values.iloc[np.argsort(cluster_size_lookup)].transpose(), fmt='',  # fmt="f",
                    robust=True, cmap='RdBu_r', vmin=color_scale_min, vmax=color_scale_max,
                    cbar_kws={'label': color_scale_title})
        plt.yticks(rotation=-30, va='bottom')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        # plt.title('Cluster Heatmap')

    @staticmethod
    def kmeans_elbow_plot(data: pd.DataFrame,
                          num_clusters: list,
                          transformations: List[TransformerBase]=None):
        scores = []
        for cluster in num_clusters:
            fitter = ModelFitter(model=ClusteringKMeans(),
                                 model_transformations=None if transformations is None else
                                 [x.clone() for x in transformations])
            fitter.fit(data=data, hyper_params=ClusteringKMeansHP(num_clusters=cluster))
            # noinspection PyUnresolvedReferences
            scores.append(abs(fitter.model.score))

        plt.plot(num_clusters, scores, linestyle='--', marker='o', color='b')
        plt.xlabel('K')
        plt.ylabel('SSE')
        plt.title('SSE vs. K')
