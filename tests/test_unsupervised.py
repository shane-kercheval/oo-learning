import os
import shutil
from math import isclose

import numpy as np
from sklearn.metrics import adjusted_rand_score

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic, PyUnresolvedReferences
class UnsupervisedTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_ModelFitter_callback(self):
        # make sure that the ModelTrainer->train_callback works, other tests rely on it to work correctly.
        data = TestHelper.get_iris_data()

        # noinspection PyUnusedLocal
        def fit_callback(data_x, hyper_params):
            raise NotImplementedError()

        model_fitter = ModelFitter(model=ClusteringKMeans(),
                                   model_transformations=[CenterScaleTransformer()],
                                   fit_callback=fit_callback)

        # should raise an error from the callback definition above
        self.assertRaises(NotImplementedError,
                          lambda: model_fitter.fit(data=data, hyper_params=ClusteringKMeansHP()))

    def test_ModelFitter_persistence(self):
        # make sure that the ModelTrainer->train_callback works, other tests rely on it to work correctly.
        data = TestHelper.get_iris_data()
        data = data.drop(columns='species')

        cache_directory = TestHelper.ensure_test_directory('temp')
        assert os.path.isdir(cache_directory) is False

        model_fitter = ModelFitter(model=ClusteringKMeans(),
                                   model_transformations=[CenterScaleTransformer()],
                                   persistence_manager=LocalCacheManager(cache_directory=cache_directory))
        clusters = model_fitter.fit_predict(data=data, hyper_params=ClusteringKMeansHP())

        expected_file = os.path.join(cache_directory, 'ClusteringKMeans_num_clusters_8_init_method_k-means_num_different_seeds_10_max_iterations_300_precompute_distances_auto_algorithm_auto.pkl')  # noqa
        assert os.path.isfile(expected_file)

        # use a different seed which was verified to produce different clusters, but since we are getting
        # the cached file, we should retrieve the model with the old seed which should produce the original
        # clusters (i.e. commenting out `persistence_manager=...` was verified to produce different
        # `new_clusters`
        new_model_fitter = ModelFitter(model=ClusteringKMeans(seed=123),
                                       model_transformations=[CenterScaleTransformer()],
                                       persistence_manager=LocalCacheManager(cache_directory=cache_directory))
        new_clusters = new_model_fitter.fit_predict(data=data, hyper_params=ClusteringKMeansHP())
        shutil.rmtree(cache_directory)
        # noinspection PyTypeChecker
        assert all(new_clusters == clusters)

    def test_ModelFitter_transformations(self):
        data = TestHelper.get_iris_data()
        data = data.drop(columns='species')

        transformations = [CenterScaleTransformer()]

        # NOTE: the data is shuffled within `ClusteringKMeans.train()`, but that happens after the callback
        # noinspection PyUnusedLocal
        def fit_callback(transformed_data, hyper_params):
            # make sure the data that was trained was as expected
            center_scale = CenterScaleTransformer()
            transformed_local_data = center_scale.fit_transform(data_x=data)

            # noinspection PyTypeChecker
            assert all(transformed_data.columns.values == transformed_local_data .columns.values)
            assert all(transformed_data == transformed_local_data)

        # same holdout_ratio as above
        fitter = ModelFitter(model=ClusteringKMeans(),
                              model_transformations=transformations,
                              fit_callback=fit_callback)
        fitter.fit(data=data, hyper_params=ClusteringKMeansHP())

    def test_ClusteringKMeansHP(self):
        hyper_params = ClusteringKMeansHP()
        assert hyper_params.params_dict == {'num_clusters': 8,
                                            'init_method': 'k-means++',
                                            'num_different_seeds': 10,
                                            'max_iterations': 300,
                                            'precompute_distances': 'auto',
                                            'algorithm': 'auto'}

    def test_KMeans_shuffle(self):
        data = TestHelper.get_iris_data()

        cluster_data = data.drop(columns='species')
        fitter = ModelFitter(model=ClusteringKMeans(evaluate_bss_tss=True),
                             model_transformations=[CenterScaleTransformer()])
        clusters = fitter.fit_predict(data=cluster_data, hyper_params=ClusteringKMeansHP(num_clusters=3))
        assert isclose(fitter.model.silhouette_score, 0.4589717867018717)
        assert isclose(fitter.model.score, -140.96581663074699)
        assert isclose(fitter.model.bss_tss_ratio, 0.7650569722820886)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.6201351808870379)
        num_cached = fitter.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == fitter.model.data_x_trained_head)

        # setosa == 1
        # versicolor == 0
        # virginica == 2
        lookup = ['virginica', 'setosa', 'versicolor']
        predicted_clusters = [lookup[x] for x in clusters]

        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [50, 0, 0, 50])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 39, 14, 53])
        assert all(confusion_matrix.matrix.virginica.values == [0, 11, 36, 47])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])

    def test_KMeans_no_shuffle(self):
        data = TestHelper.get_iris_data()

        cluster_data = data.drop(columns='species')
        fitter = ModelFitter(model=ClusteringKMeans(shuffle_data=False),
                             model_transformations=[CenterScaleTransformer()])
        clusters = fitter.fit_predict(data=cluster_data, hyper_params=ClusteringKMeansHP(num_clusters=3))
        assert isclose(fitter.model.silhouette_score, 0.4589717867018717)
        assert isclose(fitter.model.score, -140.96581663074699)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.6201351808870379)
        assert fitter.model.bss_tss_ratio is None

        num_cached = fitter.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == fitter.model.data_x_trained_head)

        # setosa == 1
        # versicolor == 2
        # virginica == 0
        lookup = ['virginica', 'setosa', 'versicolor']
        predicted_clusters = [lookup[x] for x in clusters]

        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [50, 0, 0, 50])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 39, 14, 53])
        assert all(confusion_matrix.matrix.virginica.values == [0, 11, 36, 47])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])

    def test_KMeans_elbow_sse(self):
        data = TestHelper.get_iris_data()
        TestHelper.check_plot('data/test_unsupervised/test_kmeans_elbow_plot.png',
                              lambda: Clustering.kmeans_elbow_sse_plot(data=data.drop(columns='species'),
                                                                       num_clusters=list(range(1, 9)),
                                                                       transformations=[CenterScaleTransformer()]))  # noqa

    def test_KMeans_elbow_bss_tss(self):
        data = TestHelper.get_iris_data()
        TestHelper.check_plot('data/test_unsupervised/test_kmeans_elbow_plot_bss_tss.png',
                              lambda: Clustering.kmeans_elbow_bss_tss_plot(data=data.drop(columns='species'),
                                                                           num_clusters=list(range(1, 9)),
                                                                           transformations=[CenterScaleTransformer()]))  # noqa

    def test_KMeans_heatmap(self):
        data = TestHelper.get_iris_data()
        # remove setosa, in order to test 2 clusters with same amount of data, so we can verify axis on graph
        num_drop = 3
        data = data.drop(index=list(range(num_drop)))

        data.loc[num_drop, 'sepal_length'] = np.nan
        data.loc[num_drop+1, 'petal_length'] = np.nan

        assert data.isnull().sum().sum() == 2

        # impute missing values by species, then center/scale, then remove the species column
        transformations = [ImputationTransformer(group_by_column='species'),
                           RemoveColumnsTransformer(columns=['species']),
                           CenterScaleTransformer()]

        fitter = ModelFitter(model=ClusteringKMeans(seed=199),
                             model_transformations=transformations)
        clusters = fitter.fit_predict(data=data, hyper_params=ClusteringKMeansHP(num_clusters=3))
        assert data.isnull().sum().sum() == 2  # make sure original data wasn't transformed

        assert all(np.bincount(np.array(clusters)) == [47, 47, 53])  # make sure 2 clusters are same size
        assert len(clusters) == len(data)
        # CENTER/SCALE
        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean_min_max.png',  # noqa
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30,
                                  color_scale_min=-3,
                                  color_scale_max=3))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_actual_mean.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,  # noqa
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.ACTUAL,
                                  y_axis_rotation=30))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_actual_median.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,  # noqa
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEDIAN,
                                  display_values=ClusteringHeatmapValues.ACTUAL,
                                  y_axis_rotation=30))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_median.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,  # noqa
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEDIAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30))
        # PERCENTILES
        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_percentiles_strategy_mean.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.PERCENTILES,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_percentiles_actual_mean.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,  # noqa
                                  trans_strategy=ClusteringHeatmapTransStrategy.PERCENTILES,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.ACTUAL,
                                  y_axis_rotation=30))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_percentiles_actual_median.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,  # noqa
                                  trans_strategy=ClusteringHeatmapTransStrategy.PERCENTILES,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEDIAN,
                                  display_values=ClusteringHeatmapValues.ACTUAL,
                                  y_axis_rotation=30))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_percentiles_strategy_median.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,  # noqa
                                  trans_strategy=ClusteringHeatmapTransStrategy.PERCENTILES,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEDIAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30))

    def test_hierarchical_dendogram_plot(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        TestHelper.check_plot('data/test_unsupervised/test_hierarchical_dendogram_plot.png',
                              lambda: Clustering.hierarchical_dendogram_plot(data=cluster_data,
                                                                             transformations=[NormalizationVectorSpaceTransformer()],  # noqa
                                                                             linkage=ClusteringHierarchicalLinkage.WARD))  # noqa

    def test_Hierarchical_sklearn_normalization(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        fitter = ModelFitter(model=ClusteringHierarchical(),
                             model_transformations=[NormalizationVectorSpaceTransformer()],
                             )
        clusters = fitter.fit_predict(data=cluster_data,
                                      hyper_params=ClusteringHierarchicalHP(num_clusters=3))  # noqa
        assert isclose(fitter.model.silhouette_score, 0.556059949257158)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.8856970310281228)

    def test_Hierarchical_no_shuffle(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')

        fitter = ModelFitter(model=ClusteringHierarchical(shuffle_data=False),
                             model_transformations=[NormalizationTransformer()],
                             )
        clusters = fitter.fit_predict(data=cluster_data, hyper_params=ClusteringHierarchicalHP(num_clusters=3))  # noqa
        assert isclose(fitter.model.silhouette_score, 0.5043490792923951)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.7195837484778037)

        num_cached = fitter.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == fitter.model.data_x_trained_head)

        # setosa == 1
        # versicolor == 0
        # virginica == 2
        lookup = ['versicolor', 'setosa', 'virginica']
        predicted_clusters = [lookup[x] for x in clusters]
        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [50, 0, 0, 50])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 50, 17, 67])
        assert all(confusion_matrix.matrix.virginica.values == [0, 0, 33, 33])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])

    def test_Hierarchical_shuffle(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')

        fitter = ModelFitter(model=ClusteringHierarchical(shuffle_data=True),
                             model_transformations=[NormalizationTransformer()],
                             )
        clusters = fitter.fit_predict(data=cluster_data,
                                      hyper_params=ClusteringHierarchicalHP(num_clusters=3))  # noqa
        assert isclose(fitter.model.silhouette_score, 0.5043490792923951)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.7195837484778037)

        num_cached = fitter.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == fitter.model.data_x_trained_head)

        # setosa == 1
        # versicolor == 0
        # virginica == 2
        lookup = ['versicolor', 'setosa', 'virginica']
        predicted_clusters = [lookup[x] for x in clusters]
        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [50, 0, 0, 50])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 50, 17, 67])
        assert all(confusion_matrix.matrix.virginica.values == [0, 0, 33, 33])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])

    def test_Hierarchical_shuffle_string_index(self):
        # noinspection SpellCheckingInspection
        """
        because AgglomerativeClustering doesn't have a `predict`, ClusteringHierarchical has some additional
        logic to "un-shuffle" the data in `predict()` method, need to test string indexes as well as integer
        """
        data = TestHelper.get_iris_data()
        data.index = data.index.astype(str)
        assert all([isinstance(x, str) for x in data.index.values])
        cluster_data = data.drop(columns='species')

        fitter = ModelFitter(model=ClusteringHierarchical(shuffle_data=True),
                             model_transformations=[NormalizationTransformer()],
                             )
        clusters = fitter.fit_predict(data=cluster_data,
                                      hyper_params=ClusteringHierarchicalHP(num_clusters=3))  # noqa
        assert isclose(fitter.model.silhouette_score, 0.5043490792923951)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.7195837484778037)

        num_cached = fitter.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == fitter.model.data_x_trained_head)

        # setosa == 1
        # versicolor == 0
        # virginica == 2
        lookup = ['versicolor', 'setosa', 'virginica']
        predicted_clusters = [lookup[x] for x in clusters]
        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [50, 0, 0, 50])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 50, 17, 67])
        assert all(confusion_matrix.matrix.virginica.values == [0, 0, 33, 33])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])

    def test_DBSCAN_shuffle(self):
        # after normalization, default epsilon of 0.5 is too small
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        fitter = ModelFitter(model=ClusteringDBSCAN(), model_transformations=[NormalizationTransformer()])
        clusters = fitter.fit_predict(data=cluster_data, hyper_params=ClusteringDBSCANHP())
        assert isclose(fitter.model.silhouette_score, -1)
        assert isclose(adjusted_rand_score(data.species, clusters), 0)

        # try smaller epsilon
        fitter = ModelFitter(model=ClusteringDBSCAN(), model_transformations=[NormalizationTransformer()])
        clusters = fitter.fit_predict(data=cluster_data, hyper_params=ClusteringDBSCANHP(epsilon=0.25))
        assert isclose(fitter.model.silhouette_score, 0.5759307352949353)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.5557898627256278)

        num_cached = fitter.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == fitter.model.data_x_trained_head)

        # setosa == 1
        # versicolor == 0
        # virginica == -1
        lookup = {1: 'setosa', 0: 'versicolor', -1: 'virginica'}
        predicted_clusters = [lookup[x] for x in clusters]

        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [49, 0, 0, 49])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 50, 49, 99])
        assert all(confusion_matrix.matrix.virginica.values == [1, 0, 1, 2])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])

    # noinspection PyUnresolvedReferences
    def test_DBSCAN_no_shuffle(self):
        # after normalization, default epsilon of 0.5 is too small
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        fitter = ModelFitter(model=ClusteringDBSCAN(shuffle_data=False),
                             model_transformations=[NormalizationTransformer()])
        clusters = fitter.fit_predict(data=cluster_data, hyper_params=ClusteringDBSCANHP())
        assert isclose(fitter.model.silhouette_score, -1)
        assert isclose(adjusted_rand_score(data.species, clusters), 0)

        # try smaller epsilon
        fitter = ModelFitter(model=ClusteringDBSCAN(), model_transformations=[NormalizationTransformer()])
        clusters = fitter.fit_predict(data=cluster_data, hyper_params=ClusteringDBSCANHP(epsilon=0.25))
        assert isclose(fitter.model.silhouette_score, 0.5759307352949353)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.5557898627256278)

        num_cached = fitter.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == fitter.model.data_x_trained_head)

        lookup = {1: 'setosa', 0: 'versicolor', -1: 'virginica'}
        predicted_clusters = [lookup[x] for x in clusters]

        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [49, 0, 0, 49])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 50, 49, 99])
        assert all(confusion_matrix.matrix.virginica.values == [1, 0, 1, 2])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])
