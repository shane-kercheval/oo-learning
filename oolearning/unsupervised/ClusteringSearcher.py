import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import List, Union

from oolearning.evaluators.ScoreClusteringBase import ScoreClusteringBase
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_processors.ModelTrainer import ModelTrainer
from oolearning.transformers.NormalizationTransformer import NormalizationTransformer
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class ClusteringSearcherResult:
    """
    Contains the information for a single model (i.e. ModelInfo) that clustered with >=1 hyper-parameter
        combinations.
    """
    def __init__(self, model_description: str,
                 hyper_params_grid: pd.DataFrame,
                 scores: List[List[ScoreClusteringBase]],
                 resulting_num_clusters: List[int]):
        """

        :param model_description:
        :param hyper_params_grid:
        :param scores: each key is a h
        """
        self._model_description = model_description
        self._hyper_params_grid = hyper_params_grid
        self._scores = scores
        self._resulting_num_clusters = resulting_num_clusters

    @property
    def model_description(self):
        return self._model_description

    @property
    def hyper_params_grid(self):
        return self._hyper_params_grid

    @property
    def scores(self):
        return self._scores

    @property
    def resulting_num_clusters(self):
        return self._resulting_num_clusters


class ClusteringSearcherResults:
    def __init__(self, searcher_results: List[ClusteringSearcherResult]):
        self._searcher_results = searcher_results
        self._all_model_results = None

        # get the score names, will be the same for every result/iteration, so grab the first iteration/result
        self._score_column_names = [x.name for x in searcher_results[0].scores[0]]
        for result in searcher_results:
            # build up a pandas DataFrame that has a row for each Model/hyper-params-combo, and corresponding
            # scores
            param_grid = result.hyper_params_grid
            description = result.model_description
            results_df = pd.DataFrame(
                {'model': description,
                 # convert hyper_parameters to list of dictionaries (and store in column)
                 'hyper_params': [param_grid.iloc[x].to_dict() for x in range(len(param_grid))],
                 'Num. Clusters': result.resulting_num_clusters}
            )
            results_df = results_df[['model', 'hyper_params', 'Num. Clusters']]

            num_score_objects = len(result.scores[0])  # look at length of first element
            for score_index in range(num_score_objects):
                score_objects = [x[score_index] for x in result.scores]
                results_df[self.score_names[score_index]] = [x.value for x in score_objects]

            results_df['Max Score'] = results_df[self.score_names].max(axis=1).values

            self._all_model_results = results_df if self._all_model_results is None \
                else pd.concat([self._all_model_results, results_df])

    @property
    def results(self):
        return self._all_model_results

    @property
    def score_names(self):
        return self._score_column_names

    def heatmap(self, figure_size: set=(12, 12)):
        heatmap_columns = ['Num. Clusters'] + self.score_names + ['Max Score']
        heat_map_data = self.results[heatmap_columns]
        heat_map_data.index = ["{} - {}".format(x, str(y)) for x, y in zip(self.results.model.values,
                                                                           self.results.hyper_params.values)]  # noqa
        plt.figure(figsize=figure_size)
        # fig = plt.gcf()
        # fig.set_size_inches(12, 12)
        sns.heatmap(NormalizationTransformer().fit_transform(heat_map_data),  # normalize based on columns
                    annot=round(heat_map_data, 3), fmt='',  # fmt="f",
                    robust=True, cmap='RdBu_r',  # vmin=color_scale_min, vmax=color_scale_max,
                    cbar_kws={'label': 'Normalized Score (Per Column)'},
                    yticklabels=heat_map_data.index.values
                    )
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()


class ClusteringSearcher:
    """
    TODO: DOCUMENT
    """
    def __init__(self,
                 model_infos: List[ModelInfo],
                 scores: List[ScoreClusteringBase],
                 global_transformations: Union[List[TransformerBase], None] = None,
                 # model_persistence_manager: PersistenceManagerBase = None,
                 # parallelization_cores: int=-1
                 ):

        self._models = [x.model for x in model_infos]
        self._model_descriptions = [x.description for x in model_infos]

        # ensure all descriptions are unique
        # length of unique values (via `set()`) should be the same as the length of all the values
        assert len(set(self._model_descriptions)) == len(self._model_descriptions)

        self._model_transformations = [x.transformations for x in model_infos]
        self._model_hyper_params_objects = [x.hyper_params for x in model_infos]
        self._model_hyper_params_grids = [x.hyper_params_grid for x in model_infos]
        self._scores = scores
        self._global_transformations = global_transformations

    def search(self, data: pd.DataFrame):
        data = data.copy()
        if self._global_transformations is not None:
            # noinspection PyTypeChecker
            data = TransformerPipeline(transformations=self._global_transformations).\
                fit_transform(data_x=data)

        searcher_results = []
        for model_index in range(len(self._models)):
            local_model = self._models[model_index]
            local_model_description = self._model_descriptions[model_index]
            local_transformations = self._model_transformations[model_index]
            local_hyper_params_object = self._model_hyper_params_objects[model_index]
            local_hyper_params_grid = self._model_hyper_params_grids[model_index]

            # could pass transformations to fitter, but since we are iterating, let's just transform once
            if local_transformations is not None:
                data = TransformerPipeline(transformations=[x.clone() for x in local_transformations]).\
                    fit_transform(data_x=data)

            score_results = []
            resulting_num_clusters = []
            params_grid = local_hyper_params_grid.params_grid
            for params_index in range(len(params_grid)):
                trainer = ModelTrainer(model=local_model.clone(),
                                       scores=[x.clone() for x in self._scores])
                # get current hyper parameter combination
                hyper_params = local_hyper_params_object.clone()
                hyper_params.update_dict(params_grid.iloc[params_index].to_dict())
                clusters = trainer.train_predict_eval(data=data, hyper_params=hyper_params)
                score_results.append(trainer.training_scores)  # appending list to list
                resulting_num_clusters.append(len(np.unique(clusters)))  # get number of unique clusters

            # noinspection PyTypeChecker
            searcher_results.append(ClusteringSearcherResult(model_description=local_model_description,
                                                             hyper_params_grid=params_grid,
                                                             scores=score_results,
                                                             resulting_num_clusters=resulting_num_clusters))

        return ClusteringSearcherResults(searcher_results=searcher_results)
