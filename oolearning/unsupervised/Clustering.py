from enum import Enum, unique
from scipy.cluster import hierarchy
from typing import List, Union
# from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count
from multiprocessing import get_context
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns

from oolearning.model_processors.ModelTrainer import ModelTrainer
from oolearning.transformers.TransformerPipeline import TransformerPipeline
from oolearning.transformers.CenterScaleTransformer import CenterScaleTransformer
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.unsupervised.ClusteringHierarchical import ClusteringHierarchicalLinkage
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


def single_kmeans(args):
    data = args['data']
    transformations = args['transformations']
    num_clusters = args['num_clusters']
    trainer = ModelTrainer(model=ClusteringKMeans(),
                           model_transformations=None if transformations is None else
                           [x.clone() for x in transformations])
    trainer.train_predict_eval(data=data, hyper_params=ClusteringKMeansHP(num_clusters=num_clusters))

    # noinspection PyUnresolvedReferences
    return abs(trainer.model.score)


class Clustering:

    @staticmethod
    def cluster_heatmap(data: pd.DataFrame,
                        clusters: np.array,
                        trans_strategy: ClusteringHeatmapTransStrategy.CENTER_SCALE,
                        agg_strategy: ClusteringHeatmapAggStrategy.MEAN,
                        display_values: ClusteringHeatmapValues.ACTUAL,
                        color_scale_min: Union[int, float, None]=None,
                        color_scale_max: Union[int, float, None] = None,
                        color_map: str='RdBu_r',
                        plot_size: tuple=(7, 7),
                        axis_font_size: int=10,
                        annotation_font_size: int=10,
                        y_axis_rotation: int=0,
                        round_by: int=2,
                        round_by_custom: Union[dict, None]=None):
        """
        :param data: data from which the clusters were generated
            It is recommended you use the pre-transformed data (i.e. could contain missing values), otherwise
                mean/median/etc. is calculated on filled values (i.e. replacing NAs with 0) which will change
                the aggregated values.

                e.g. if original column was `(0, 1, 0, 1, NA, NA)` and transformed column was
                    `(0, 1, 0, 1, 0, 0)` then the original mean is `0.5` while the mean for the transformed
                    column is `0.333`
        :param clusters: the cluster number for each row in `data`

        :param agg_strategy: This strategy (ClusteringHeatmapAggStrategy) will determine how the values for
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

            If set to `None`, then no underlying transformation happens and there is no difference between
                `display_values` of `ClusteringHeatmapValues.STRATEGY` and `ClusteringHeatmapValues.ACTUAL`.
                Also, the `color_scale_min` and `color_scale_max` parameters are defaulted to the min/max
                of the values in the dataset.

        :param color_scale_min: min value for the color scale
        :param color_scale_max: max value for the color scale
        :param color_map: "matplotlib colormap name or object, or list of colors, optional
            The mapping from data values to color space. If not provided, the default will depend on whether
            ``center`` is set."
        :param plot_size: size of the plot (width, height)
        :param axis_font_size: the font size for the axis labels
        :param annotation_font_size: the font size for the numbers (i.e. annotation) inside the cells
        :param y_axis_rotation: degrees to rotate the y-axis labels
        :param round_by: the number of decimals to round aggregated (i.e. cell) values by
        :param round_by_custom: a dictionary containing the feature to round as the key, and the value to
            round by as the column. `round_by_custom` takes priority over `round_by`. Any column not specified
            will be rounded by `round_by`.
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

        if trans_strategy is None:
            # if there is no transformation, then default the min/max scale to the min/max of the data
            transformed_data = data
            if color_scale_min is None:
                color_scale_min = data.min().min()
            if color_scale_max is None:
                color_scale_max = data.max().max()
            color_scale_title = ""
        elif trans_strategy == ClusteringHeatmapTransStrategy.CENTER_SCALE:
            # same as getting the cluster centers if using the underlying sklearn model, if STRATEGY was used:
            # cluster_centers = pd.DataFrame(fitter.model.model_object.cluster_centers_)
            # cluster_centers.columns = columns_to_keep
            transformed_data = CenterScaleTransformer().fit_transform(data)
            if color_scale_min is None:
                color_scale_min = -2
            if color_scale_max is None:
                color_scale_max = 2
            color_scale_title = "Cluster's {} Stan. Dev. (Z-Score) of Feature".format(agg_strategy_label)

        elif trans_strategy == ClusteringHeatmapTransStrategy.PERCENTILES:
            transformed_data = data.rank(pct=True)
            if color_scale_min is None:
                color_scale_min = 0
            if color_scale_max is None:
                color_scale_max = 1
            color_scale_title = "Cluster's {} Percentile Value for the Feature".format(agg_strategy_label)
        else:
            raise NotImplementedError()

        transformed_data['cluster'] = clusters

        cluster_size = transformed_data.groupby('cluster').size()
        cluster_size_lookup = cluster_size.values

        group_data = transformed_data.groupby('cluster').apply(agg_method).drop(columns='cluster')
        # noinspection PyUnresolvedReferences
        indexes_with_sizes = ['{1} - ~{2}% ({0})'.format(index, size, int(round(size/len(clusters) * 100, 0)))
                              for index, size in zip(group_data.index.values, cluster_size_lookup)]
        group_data.index = indexes_with_sizes

        if display_values == ClusteringHeatmapValues.ACTUAL:
            temp = data
            temp['cluster'] = clusters
            values = temp.groupby('cluster').apply(agg_method).drop(columns='cluster')
            values.index = indexes_with_sizes

        elif display_values == ClusteringHeatmapValues.STRATEGY:
            values = group_data.copy()

        else:
            raise NotImplementedError()

        if round_by_custom is None:
            values = values.round(round_by)
        else:
            for column in values.columns.values:
                if column in round_by_custom:
                    round_by_temp = round_by_custom[column]
                    if round_by_temp == 0:
                        values[column] = values[column].round(0).astype(int)
                    else:
                        values[column] = values[column].round(round_by_temp)
                else:
                    values[column] = values[column].round(round_by)

        for column in values.columns.values:
            values[column] = values[column].astype(str)

        sns.heatmap(group_data.iloc[np.argsort(cluster_size_lookup)].transpose(),
                    annot=values.iloc[np.argsort(cluster_size_lookup)].transpose(),
                    annot_kws={"size": annotation_font_size},
                    fmt='',  # fmt="f",
                    robust=True, cmap=color_map, vmin=color_scale_min, vmax=color_scale_max,
                    cbar_kws={'label': color_scale_title},
                    xticklabels=True, yticklabels=True)
        plt.yticks(rotation=y_axis_rotation * -1,
                   va='center' if y_axis_rotation == 0 else 'bottom',
                   fontsize=axis_font_size)
        plt.xticks(rotation=30, ha='right', fontsize=axis_font_size)
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
        plt.tight_layout()
        # plt.title('Cluster Heatmap')

    @staticmethod
    def kmeans_elbow_sse_plot(data: pd.DataFrame,
                              num_clusters: list,
                              transformations: List[TransformerBase]=None,
                              parallelization_cores: int = -1):

        single_kmeans_args = [dict(data=data,
                                   transformations=transformations,
                                   num_clusters=x)
                              for x in num_clusters]

        # if parallelization_cores == 0 or parallelization_cores == 1:
        #     results = list(map(single_kmeans, single_kmeans_args))
        # else:
        #     cores = cpu_count() if parallelization_cores == -1 else parallelization_cores
        #     # with ThreadPool(cores) as pool:
        #     # https://codewithoutrules.com/2018/09/04/python-multiprocessing/
        #     with get_context("spawn").Pool(cores) as pool:
        #         results = list(pool.map(single_kmeans, single_kmeans_args))
        results = list(map(single_kmeans, single_kmeans_args))

        scores = [x for x in results]

        plt.plot(num_clusters, scores, linestyle='--', marker='o', color='b')
        plt.gcf().set_size_inches(9, 6)
        plt.xlabel('K')
        plt.ylabel('SSE')
        plt.title('SSE vs. K')

    @staticmethod
    def kmeans_elbow_bss_tss_plot(data: pd.DataFrame,
                                  num_clusters: list,
                                  transformations: List[TransformerBase]=None):
        """
        Method from https://chih-ling-hsu.github.io/2018/01/02/clustering-python
        """
        ratios = []
        for cluster in num_clusters:
            trainer = ModelTrainer(model=ClusteringKMeans(evaluate_bss_tss=True),
                                   model_transformations=None if transformations is None else
                                   [x.clone() for x in transformations])
            trainer.train_predict_eval(data=data, hyper_params=ClusteringKMeansHP(num_clusters=cluster))
            # noinspection PyUnresolvedReferences
            ratios.append(abs(trainer.model.bss_tss_ratio))

        plt.plot(num_clusters, ratios, linestyle='--', marker='o', color='b')
        plt.gcf().set_size_inches(9, 6)
        plt.xlabel('K')
        plt.ylabel('BSS/TSS RATIO')
        plt.title('BSS/TSS RATIO vs. K')

    @staticmethod
    def silhouette_stats(clustered_data: pd.DataFrame, clusters: np.array) -> pd.DataFrame:
        return pd.Series(silhouette_samples(clustered_data, clusters)).groupby(clusters).describe()

    @staticmethod
    def silhouette_plot(clustered_data: pd.DataFrame, clusters: np.array, figure_size: tuple=(10, 10)):
        """
        modified from
            http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
        :param clustered_data: this is the transformed data that the cluster algorithm used to cluster.
        :param clusters: the resulting clusters.
        :param figure_size: set (width, height) in inches.
        :return:
        """
        # Create a subplot with 1 row and 1 columns
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(figure_size[0], figure_size[1])

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        n_clusters = len(set(clusters))
        ax1.set_ylim([0, len(clustered_data) + (n_clusters + 1) * 10])

        cluster_labels = clusters

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(clustered_data, cluster_labels)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(clustered_data, cluster_labels)

        y_lower = 10
        for i in np.unique(clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            # noinspection PyUnresolvedReferences
            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        plt.suptitle(("Silhouette analysis for clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    @staticmethod
    def hierarchical_dendogram_plot(data: pd.DataFrame,
                                    transformations: List[TransformerBase] = None,
                                    linkage: ClusteringHierarchicalLinkage=ClusteringHierarchicalLinkage.WARD,
                                    color_threshold=None,
                                    figure_size: tuple=(22, 18),
                                    ):
        """

        :param data: dataset to cluster on
        :param transformations: transformations to apply before clustering
        :param linkage: the type of clustering to apply
        :param color_threshold: the value of the y-axis to apply the 'horizontal cutoff' for clustering.
            You'll likely need to first use `None`, then visualize a more appropriate value based off of the
            dendogram.

            For a more precise description, see
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
        :param figure_size: size of the figure
        """
        transformed_data = TransformerPipeline(transformations=transformations).fit_transform(data)
        # Specify the linkage type. Scipy accepts 'ward', 'complete', 'average', as well as other values
        linkage_matrix = hierarchy.linkage(transformed_data.values, linkage.value)
        plt.figure(figsize=figure_size)
        hierarchy.dendrogram(linkage_matrix, color_threshold=color_threshold)
