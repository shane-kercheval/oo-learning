import os
import shutil
import hdbscan
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

    def test_ModelTrainer_callback(self):
        # make sure that the ModelTrainer->train_callback works, other tests rely on it to work correctly.
        data = TestHelper.get_iris_data()

        # noinspection PyUnusedLocal
        def fit_callback(data_x, target, hyper_params):
            raise NotImplementedError()

        model_trainer = ModelTrainer(model=ClusteringKMeans(),
                                     model_transformations=[CenterScaleTransformer()],
                                     train_callback=fit_callback)

        # should raise an error from the callback definition above
        self.assertRaises(NotImplementedError,
                          lambda: model_trainer.train_predict_eval(data=data,
                                                                   hyper_params=ClusteringKMeansHP()))

    def test_ModelTrainer_persistence(self):
        # make sure that the ModelTrainer->train_callback works, other tests rely on it to work correctly.
        data = TestHelper.get_iris_data()
        data = data.drop(columns='species')

        cache_directory = TestHelper.ensure_test_directory('temp')
        assert os.path.isdir(cache_directory) is False

        model_fitter = ModelTrainer(model=ClusteringKMeans(),
                                    model_transformations=[CenterScaleTransformer()],
                                    persistence_manager=LocalCacheManager(cache_directory=cache_directory))
        clusters = model_fitter.train_predict_eval(data=data, hyper_params=ClusteringKMeansHP())

        expected_file = os.path.join(cache_directory, 'ClusteringKMeans_num_clusters_8_init_method_k-means_num_different_seeds_10_max_iterations_300_precompute_distances_auto_algorithm_auto.pkl')  # noqa
        assert os.path.isfile(expected_file)

        # use a different seed which was verified to produce different clusters, but since we are getting
        # the cached file, we should retrieve the model with the old seed which should produce the original
        # clusters (i.e. commenting out `persistence_manager=...` was verified to produce different
        # `new_clusters`
        new_model_fitter = ModelTrainer(model=ClusteringKMeans(seed=123),
                                        model_transformations=[CenterScaleTransformer()],
                                        persistence_manager=LocalCacheManager(cache_directory=cache_directory))  # noqa
        new_clusters = new_model_fitter.train_predict_eval(data=data, hyper_params=ClusteringKMeansHP())
        shutil.rmtree(cache_directory)
        # noinspection PyTypeChecker
        assert all(new_clusters == clusters)

    def test_ModelTrainer_transformations(self):
        data = TestHelper.get_iris_data()
        data = data.drop(columns='species')

        transformations = [CenterScaleTransformer()]

        # noinspection PyUnusedLocal
        def fit_callback(transformed_data, target, hyper_params):
            # make sure the data that was trained was as expected
            center_scale = CenterScaleTransformer()
            transformed_local_data = center_scale.fit_transform(data_x=data)

            # noinspection PyTypeChecker
            assert all(transformed_data.columns.values == transformed_local_data .columns.values)
            assert all(transformed_data == transformed_local_data)

        # same holdout_ratio as above
        trainer = ModelTrainer(model=ClusteringKMeans(),
                               model_transformations=transformations,
                               train_callback=fit_callback)
        trainer.train_predict_eval(data=data, hyper_params=ClusteringKMeansHP())

    def test_ClusteringKMeansHP(self):
        hyper_params = ClusteringKMeansHP()
        assert hyper_params.params_dict == {'num_clusters': 8,
                                            'init_method': 'k-means++',
                                            'num_different_seeds': 10,
                                            'max_iterations': 300,
                                            'precompute_distances': 'auto',
                                            'algorithm': 'auto'}

    def test_KMeans(self):
        data = TestHelper.get_iris_data()

        cluster_data = data.drop(columns='species')
        trainer = ModelTrainer(model=ClusteringKMeans(evaluate_bss_tss=True),
                               model_transformations=[CenterScaleTransformer()],
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringKMeansHP(num_clusters=3))
        assert trainer.model.model_object.n_clusters == 3

        # make sure Score object, when manually calculated, is the expected value,
        # then verify trainer.training_score
        score = SilhouetteScore().calculate(clustered_data=CenterScaleTransformer().fit_transform(cluster_data),  # noqa
                                            clusters=clusters)
        assert isclose(score, 0.45994823920518646)
        assert isclose(score, trainer.training_scores[0].value)

        assert isclose(trainer.model.score, -139.82049635974977)
        assert isclose(trainer.model.bss_tss_ratio, 0.7669658394004155)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.6201351808870379)
        num_cached = trainer.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == trainer.model.data_x_trained_head)

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

    def test_silhouette_stats(self):
        data = TestHelper.get_iris_data()

        cluster_data = data.drop(columns='species')
        cluster_data = CenterScaleTransformer().fit_transform(cluster_data)
        trainer = ModelTrainer(model=ClusteringKMeans(evaluate_bss_tss=True),
                               model_transformations=None,
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringKMeansHP(num_clusters=3))
        assert trainer.model.model_object.n_clusters == 3

        # make sure Score object, when manually calculated, is the expected value,
        # then verify trainer.training_score
        score = SilhouetteScore().calculate(
            clustered_data=CenterScaleTransformer().fit_transform(cluster_data),  # noqa
            clusters=clusters)
        assert isclose(score, 0.45994823920518646)
        assert isclose(score, trainer.training_scores[0].value)

        assert isclose(trainer.model.score, -139.82049635974977)
        assert isclose(trainer.model.bss_tss_ratio, 0.7669658394004155)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.6201351808870379)
        silhouette_stats = Clustering.silhouette_stats(clustered_data=cluster_data, clusters=clusters)
        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_unsupervised/test_silhouette_stats.pkl'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=silhouette_stats)

    def test_silhouette_plot(self):
        data = TestHelper.get_iris_data()

        cluster_data = data.drop(columns='species')
        cluster_data = CenterScaleTransformer().fit_transform(cluster_data)
        trainer = ModelTrainer(model=ClusteringKMeans(evaluate_bss_tss=True),
                               model_transformations=None,
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringKMeansHP(num_clusters=3))
        assert trainer.model.model_object.n_clusters == 3

        # make sure Score object, when manually calculated, is the expected value,
        # then verify trainer.training_score
        score = SilhouetteScore().calculate(
            clustered_data=CenterScaleTransformer().fit_transform(cluster_data),  # noqa
            clusters=clusters)
        assert isclose(score, 0.45994823920518646)
        assert isclose(score, trainer.training_scores[0].value)

        assert isclose(trainer.model.score, -139.82049635974977)
        assert isclose(trainer.model.bss_tss_ratio, 0.7669658394004155)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.6201351808870379)

        TestHelper.check_plot('data/test_unsupervised/test_silhouette_plot.png',
                              lambda: Clustering.silhouette_plot(clustered_data=cluster_data,
                                                                 clusters=clusters),
                              set_size_w_h=None)
        TestHelper.check_plot('data/test_unsupervised/test_silhouette_plot_size.png',
                              lambda: Clustering.silhouette_plot(clustered_data=cluster_data,
                                                                 clusters=clusters,
                                                                 figure_size=(8, 6)),
                              set_size_w_h=None)

    def test_KMeans_elbow_sse(self):
        data = TestHelper.get_iris_data()
        TestHelper.check_plot('data/test_unsupervised/test_kmeans_elbow_plot.png',
                              lambda: Clustering.kmeans_elbow_sse_plot(data=data.drop(columns='species'),
                                                                       num_clusters=list(range(1, 9)),
                                                                       transformations=[CenterScaleTransformer()]),  # noqa
                              set_size_w_h=None)
        data = TestHelper.get_iris_data()
        TestHelper.check_plot('data/test_unsupervised/test_kmeans_elbow_plot_not_parallelized.png',
                              lambda: Clustering.kmeans_elbow_sse_plot(data=data.drop(columns='species'),
                                                                       num_clusters=list(range(1, 9)),
                                                                       transformations=[CenterScaleTransformer()],  # noqa
                                                                       parallelization_cores=0),
                              set_size_w_h=None)

    def test_KMeans_elbow_bss_tss(self):
        data = TestHelper.get_iris_data()
        TestHelper.check_plot('data/test_unsupervised/test_kmeans_elbow_plot_bss_tss.png',
                              lambda: Clustering.kmeans_elbow_bss_tss_plot(data=data.drop(columns='species'),
                                                                           num_clusters=list(range(1, 9)),
                                                                           transformations=[CenterScaleTransformer()]),  # noqa
                              set_size_w_h=None)

    # noinspection SpellCheckingInspection
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

        trainer = ModelTrainer(model=ClusteringKMeans(seed=199),
                               model_transformations=transformations)
        clusters = trainer.train_predict_eval(data=data, hyper_params=ClusteringKMeansHP(num_clusters=3))
        assert data.isnull().sum().sum() == 2  # make sure original data wasn't transformed

        assert all(np.bincount(np.array(clusters)) == [53, 47, 47])  # make sure 2 clusters are same size
        assert len(clusters) == len(data)
        # CENTER/SCALE
        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30))

        # test rounding specific features/row rounding
        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean_round_by_3.png',  # noqa
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30,
                                  round_by=3,
                              ))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean_round_by_custom.png',  # noqa
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30,
                                  round_by=1,
                                  round_by_custom={'sepal_width': 0, 'petal_length': 2}
                              ))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean_font_size.png',  # noqa
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=0,
                                  plot_size=(8, 4),
                                  axis_font_size=5,
                                  annotation_font_size=5)
                              )

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean_min_max.png',  # noqa
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30,
                                  color_scale_min=-3,
                                  color_scale_max=3))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_centerscale_strategy_mean_min_max_colormap.png',  # noqa
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=ClusteringHeatmapTransStrategy.CENTER_SCALE,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  y_axis_rotation=30,
                                  color_scale_min=-1,
                                  color_scale_max=2,
                                  color_map='Reds')
                              )

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

        # NO transformations
        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_no_trans_actual_mean.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=None,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.ACTUAL,
                                  ))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_no_trans_actual_mean_min_max.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=None,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.ACTUAL,
                                  color_scale_min=-5,
                                  color_scale_max=10,
                                  ))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_no_trans_strategy_mean.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=None,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEAN,
                                  display_values=ClusteringHeatmapValues.STRATEGY,
                                  ))

        TestHelper.check_plot('data/test_unsupervised/test_KMeans_heatmap_no_trans_actual_median.png',
                              lambda: Clustering.cluster_heatmap(
                                  data=data.drop(columns='species'), clusters=clusters,
                                  trans_strategy=None,
                                  agg_strategy=ClusteringHeatmapAggStrategy.MEDIAN,
                                  display_values=ClusteringHeatmapValues.ACTUAL,
                                  ))

        assert 'cluster' not in data.columns.values  # make sure we are not changing the dataset

    def test_hierarchical_dendogram_plot(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        TestHelper.check_plot('data/test_unsupervised/test_hierarchical_dendogram_plot.png',
                              lambda: Clustering.hierarchical_dendogram_plot(data=cluster_data,
                                                                             transformations=[NormalizationVectorSpaceTransformer()],  # noqa
                                                                             linkage=ClusteringHierarchicalLinkage.WARD),  # noqa
                              set_size_w_h=None)

        TestHelper.check_plot('data/test_unsupervised/test_hierarchical_dendogram_plot_threshold.png',
                              lambda: Clustering.hierarchical_dendogram_plot(data=cluster_data,
                                                                             transformations=[NormalizationVectorSpaceTransformer()],  # noqa
                                                                             linkage=ClusteringHierarchicalLinkage.WARD,  # noqa
                                                                             color_threshold=0.5),
                              set_size_w_h=None)

    def test_Hierarchical_sklearn_normalization(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        trainer = ModelTrainer(model=ClusteringHierarchical(),
                               model_transformations=[NormalizationVectorSpaceTransformer()],
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringHierarchicalHP(num_clusters=3))
        score = SilhouetteScore().calculate(
            clustered_data=NormalizationVectorSpaceTransformer().fit_transform(cluster_data),
            clusters=clusters)
        assert isclose(score, 0.5562322357473719)
        assert isclose(score, trainer.training_scores[0].value)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.8856970310281228)

    def test_Hierarchical(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')

        trainer = ModelTrainer(model=ClusteringHierarchical(),
                               model_transformations=[NormalizationTransformer()],
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringHierarchicalHP(num_clusters=3))
        assert trainer.model.model_object.n_clusters == 3
        score = SilhouetteScore().calculate(
            clustered_data=NormalizationTransformer().fit_transform(cluster_data),
            clusters=clusters)
        assert isclose(score, 0.5047999262278894)
        assert isclose(score, trainer.training_scores[0].value)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.7195837484778037)

        num_cached = trainer.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == trainer.model.data_x_trained_head)

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

    def test_Hierarchical_string_index(self):
        # noinspection SpellCheckingInspection
        data = TestHelper.get_iris_data()
        data.index = data.index.astype(str)
        assert all([isinstance(x, str) for x in data.index.values])
        cluster_data = data.drop(columns='species')

        trainer = ModelTrainer(model=ClusteringHierarchical(),
                               model_transformations=[NormalizationTransformer()],
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringHierarchicalHP(num_clusters=3))
        score = SilhouetteScore().calculate(
            clustered_data=NormalizationTransformer().fit_transform(cluster_data),
            clusters=clusters)
        assert isclose(score, 0.5047999262278894)
        assert isclose(score, trainer.training_scores[0].value)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.7195837484778037)

        num_cached = trainer.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == trainer.model.data_x_trained_head)

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

    def test_DBSCAN(self):
        # after normalization, default epsilon of 0.5 is too small
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        trainer = ModelTrainer(model=ClusteringDBSCAN(),
                               model_transformations=[NormalizationTransformer()],
                               scores=[SilhouetteScore()])
        clusters = trainer.train_predict_eval(data=cluster_data, hyper_params=ClusteringDBSCANHP())
        transformed_cluster_data = NormalizationTransformer().fit_transform(cluster_data)
        score = SilhouetteScore().calculate(
            clustered_data=transformed_cluster_data,
            clusters=clusters)
        assert isclose(score, -1)
        assert isclose(score, trainer.training_scores[0].value)
        assert isclose(adjusted_rand_score(data.species, clusters), 0)

        # try smaller epsilon
        trainer = ModelTrainer(model=ClusteringDBSCAN(),
                               model_transformations=[NormalizationTransformer()],
                               scores=[SilhouetteScore(),
                                       DensityBasedClusteringValidationScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringDBSCANHP(epsilon=0.25,
                                                                              min_samples=4))
        assert trainer.model.model_object.eps == 0.25
        assert trainer.model.model_object.min_samples == 4
        score = SilhouetteScore().calculate(
            clustered_data=transformed_cluster_data,
            clusters=clusters)
        assert isclose(score, 0.5765941443131803)
        assert isclose(score, trainer.training_scores[0].value)
        assert isclose(adjusted_rand_score(data.species, clusters), 0.5557898627256278)

        assert isclose(trainer.training_scores[1].value,
                       hdbscan.validity.validity_index(X=transformed_cluster_data.values,
                                                       labels=clusters))

        num_cached = trainer.model.data_x_trained_head.shape[0]
        assert all(data.drop(columns='species').iloc[0:num_cached] == trainer.model.data_x_trained_head)

        lookup = {0: 'setosa', 1: 'versicolor', -1: 'virginica'}
        predicted_clusters = [lookup[x] for x in clusters]

        # noinspection PyTypeChecker
        confusion_matrix = ConfusionMatrix(actual_classes=data['species'].values,
                                           predicted_classes=predicted_clusters)
        assert all(confusion_matrix.matrix.setosa.values == [49, 0, 0, 49])
        assert all(confusion_matrix.matrix.versicolor.values == [0, 50, 49, 99])
        assert all(confusion_matrix.matrix.virginica.values == [1, 0, 1, 2])
        assert all(confusion_matrix.matrix.Total.values == [50, 50, 50, 150])

    def test_HDBSCAN(self):
        data = TestHelper.get_iris_data()
        cluster_data = data.drop(columns='species')
        trainer = ModelTrainer(model=ClusteringHDBSCAN(),
                               model_transformations=[NormalizationTransformer()],
                               scores=[SilhouetteScore(),
                                       DensityBasedClusteringValidationScore()])
        clusters = trainer.train_predict_eval(data=cluster_data,
                                              hyper_params=ClusteringHDBSCANHP(min_cluster_size=2,
                                                                               min_samples=5))
        assert trainer.model.model_object.min_cluster_size == 2
        assert trainer.model.model_object.min_samples == 5
        transformed_cluster_data = NormalizationTransformer().fit_transform(cluster_data)
        score = SilhouetteScore().calculate(
            clustered_data=transformed_cluster_data,
            clusters=clusters)
        assert isclose(score, 0.630047128435471)
        assert isclose(score, trainer.training_scores[0].value)
        assert trainer.training_scores[0].name == Metric.SILHOUETTE.value
        assert isclose(adjusted_rand_score(data.species, clusters), 0.5681159420289855)

        # validity_index
        assert trainer.training_scores[1].name == Metric.DENSITY_BASED_CLUSTERING_VALIDATION.value
        assert isclose(trainer.training_scores[1].value,
                       hdbscan.validity.validity_index(X=transformed_cluster_data.values,
                                                       labels=clusters))

    # noinspection SpellCheckingInspection
    def test_ClusteringSearcher(self):
        data = TestHelper.get_iris_data()

        center_scaled_data = CenterScaleTransformer().fit_transform(data.drop(columns='species'))
        normalized_data = NormalizationTransformer().fit_transform(data.drop(columns='species'))

        model_infos = [
            ModelInfo(model=ClusteringKMeans(),
                      description='KMeans - CenterScale',
                      transformations=[CenterScaleTransformer()],
                      hyper_params=ClusteringKMeansHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'num_clusters': [2, 3, 4, 5, 6, 7]})),
            ModelInfo(model=ClusteringDBSCAN(),
                      description='DBSCAN - Normalization',
                      transformations=[NormalizationTransformer()],
                      hyper_params=ClusteringDBSCANHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'epsilon': [0.1, 0.15, 0.2, 0.25, 0.3],
                                                                     'min_samples': [3, 4, 5, 6]})),
        ]

        searcher = ClusteringSearcher(
            model_infos=model_infos,
            scores=[SilhouetteScore(),
                    DensityBasedClusteringValidationScore()],
            global_transformations=[RemoveColumnsTransformer(columns=['species'])],
        )
        results = searcher.search(data=data)
        assert results.score_names == [Metric.SILHOUETTE.value, Metric.DENSITY_BASED_CLUSTERING_VALIDATION.value]  # noqa
        ######################################################################################################
        # verify k-means scores
        ######################################################################################################
        silhouettes = []
        dbcvs = []
        kmeans_num_clusters = model_infos[0].hyper_params_grid.params_grid.num_clusters
        for num_clusters in kmeans_num_clusters:
            trainer = ModelTrainer(model=ClusteringKMeans(),
                                   scores=[SilhouetteScore(),
                                           DensityBasedClusteringValidationScore()])
            trainer.train_predict_eval(center_scaled_data,
                                       hyper_params=ClusteringKMeansHP(num_clusters=num_clusters))
            silhouettes.append(trainer.training_scores[0].value)
            dbcvs.append(trainer.training_scores[1].value)

        for index in range(len(model_infos[0].hyper_params_grid.params_grid)):
            assert isclose(results.results.iloc[index].silhouette, silhouettes[index])
            assert isclose(results.results.iloc[index].DBCV, dbcvs[index])

        ######################################################################################################
        # verify DBSCAN scores
        ######################################################################################################
        silhouettes = []
        dbcvs = []
        starting_index = len(kmeans_num_clusters)
        dbscan_params_grid = model_infos[1].hyper_params_grid.params_grid
        for index in range(len(dbscan_params_grid)):
            epsilon = dbscan_params_grid.iloc[index].epsilon
            min_samples = dbscan_params_grid.iloc[index].min_samples
            trainer = ModelTrainer(model=ClusteringDBSCAN(),
                                   scores=[SilhouetteScore(),
                                           DensityBasedClusteringValidationScore()])
            trainer.train_predict_eval(normalized_data,
                                       hyper_params=ClusteringDBSCANHP(epsilon=epsilon,
                                                                       min_samples=min_samples))
            silhouettes.append(trainer.training_scores[0].value)
            dbcvs.append(trainer.training_scores[1].value)

        for index in range(len(dbscan_params_grid)):
            assert isclose(results.results.iloc[starting_index + index].silhouette, silhouettes[index])
            assert isclose(results.results.iloc[starting_index + index].DBCV, dbcvs[index])

        file = os.path.join(os.getcwd(), TestHelper.ensure_test_directory('data/test_unsupervised/UnsupervisedSearcher.pkl'))  # noqa
        TestHelper.ensure_all_values_equal_from_file(file=file, expected_dataframe=results.results)

    def test_ClusteringSearcher_duplicate_descriptions(self):
        model_infos = [
            ModelInfo(model=ClusteringKMeans(),
                      description='1',
                      transformations=[CenterScaleTransformer()],
                      hyper_params=ClusteringKMeansHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'num_clusters': [2, 3, 4, 5, 6, 7]})),
            ModelInfo(model=ClusteringDBSCAN(),
                      description='1',
                      transformations=[NormalizationTransformer()],
                      hyper_params=ClusteringDBSCANHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'epsilon': [0.1, 0.15, 0.2, 0.25, 0.3],
                                                                     'min_samples': [3, 4, 5, 6]})),
        ]
        self.assertRaises(AssertionError,
                          lambda: ClusteringSearcher(model_infos=model_infos,
                                                     scores=[SilhouetteScore(),
                                                             DensityBasedClusteringValidationScore()],
                                                     global_transformations=[RemoveColumnsTransformer(columns=['species'])])  # noqa
        )

    def test_ClusteringSearcher_plots(self):
        data = TestHelper.get_iris_data()

        model_infos = [
            ModelInfo(model=ClusteringKMeans(),
                      description='KMeans - CenterScale',
                      transformations=[CenterScaleTransformer()],
                      hyper_params=ClusteringKMeansHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'num_clusters': [2, 3, 4, 5, 6, 7]})),
            ModelInfo(model=ClusteringDBSCAN(),
                      description='DBSCAN - Normalization',
                      transformations=[NormalizationTransformer()],
                      hyper_params=ClusteringDBSCANHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'epsilon': [0.1, 0.15, 0.2, 0.25, 0.3],
                                                                     'min_samples': [3, 4, 5, 6]})),
            ModelInfo(model=ClusteringDBSCAN(),
                      description='DBSCAN - Normalization - VC',
                      transformations=[NormalizationVectorSpaceTransformer()],
                      hyper_params=ClusteringDBSCANHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'epsilon': [0.1, 0.15, 0.2, 0.25, 0.3],
                                                                     'min_samples': [3, 4, 5, 6]})),
            ModelInfo(model=ClusteringHDBSCAN(),
                      description='HDBSCAN - Normalization',
                      transformations=[NormalizationTransformer()],
                      hyper_params=ClusteringHDBSCANHP(),
                      hyper_params_grid=HyperParamsGrid(params_dict={'min_cluster_size': [2, 3, 4, 5, 6],
                                                                     'min_samples': [3, 4, 5, 6]})),
        ]

        searcher = ClusteringSearcher(
            model_infos=model_infos,
            scores=[SilhouetteScore(),
                    DensityBasedClusteringValidationScore()],
            global_transformations=[RemoveColumnsTransformer(columns=['species'])],
        )
        results = searcher.search(data=data)
        TestHelper.check_plot('data/test_unsupervised/ClusteringSearcher_heatmap.png',
                              lambda: results.heatmap(),
                              set_size_w_h=None)
