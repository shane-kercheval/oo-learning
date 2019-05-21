from typing import List, Callable, Union

import pandas as pd

from oolearning.model_processors.GridSearchModelTuner import GridSearchModelTuner
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_processors.ModelTrainer import ModelTrainer
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.SearcherResults import SearcherResults
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.splitters.DataSplitterBase import DataSplitterBase
from oolearning.transformers.TransformerBase import TransformerBase


# noinspection SpellCheckingInspection
class ModelSearcher:
    def __init__(self,
                 # applied to all data (document fit_transform() on training and transform() on test
                 model_infos: List[ModelInfo],
                 splitter: DataSplitterBase,
                 resampler_function: Callable[[ModelWrapperBase, List[TransformerBase]], ResamplerBase],
                 global_transformations: Union[List[TransformerBase], None] = None,
                 model_persistence_manager: PersistenceManagerBase = None,
                 resampler_persistence_manager: PersistenceManagerBase = None,
                 parallelization_cores: int = -1):
        """
        A "Searcher" searches across different models and hyper-parameters (or the same models and
            hyper-parameters with different transformations, for example) with the goal of finding the "best"
            or ideal model candidates for further tuning and optimization.

        The data is split (via a Splitter) into training and holding sets. The training set will be
            used for selecting the "best" hyper parameters via (Tuner & Resampler) and then the model will be
            retrained and evaluated with selected hyper parameters with the holdout set.

        Order of Operations:
            Split Data into Training/Holdout
            For Each Model (using Training Set):
                Tune (i.e. find best hyper-params based on hyper_params_grid)
                    i.e.:
                    For each Hyper Params Combination in the Hyper Params Grid:
                        Resample
                            Do Global Transformations
                            Do Model Transformations (via `model_infos`)
                Get Best Hyper Params from Tuner object
            Retrain Best Hyper Params on All Training Data and Get Holdout Scores

        :param model_infos: modelInfo object (i.e. wraps/encapsulates model information)
        :param splitter: defines how to split the data. The training set will be used for selecting the
            "best" hyper parameters via resampling and then the model will be retrained with selected
            hyper parameters over holdout set.
        :param resampler_function: For each model, the Searcher will use a GridSearchModelTuner for selecting
            the "best" hyper parameters, using the resampler_function to define the Resampler.

            The reason we use a function rather than passing in the object itself is that not everything that
            is passed into the Resampler (at the time we are creating the Searcher) is defined at that point.
            So the Searcher passes the necessary  information/objects into the resampler_function, which then
            creates and returns a Resampler for each model when needed, also creating copies/clones of the
            models, transformations, hyper-parameters, decorators, etc., before passing in to the 
            resampler_function.
        :param global_transformations: transformations to apply to all the data, regardless of the type of
            model. For example, regardless of the model, we might want to remove the PassengerId and Name
            fields from the Titanic dataset.
        :param model_persistence_manager: a PersistenceManager defining how the underlying models should be
            cached, optional. It is closed for each model,

            When tuning, the substructure is set to "tune_[model description]" and when fitting on the entire
            dataset, the prefix is set to "final_[model description]"
        :param resampler_persistence_manager: a PersistenceManager defining how the underlying
            ResamplerResults should be cached, optional.
        :param parallelization_cores: the number of cores to use for parallelization (via underlying
            GridSearchModelTuner). -1 is all, 0 or 1 is "off"
        """
        model_descriptions = [x.description for x in model_infos]
        models = [x.model for x in model_infos]
        model_transformations = [x.transformations for x in model_infos]
        model_hyper_params_object = [x.hyper_params for x in model_infos]
        model_hyper_params_grid = [x.hyper_params_grid for x in model_infos]

        # ensure all descriptions are unique
        # length of unique values (via `set()`) should be the same as the length of all the values
        assert len(set(model_descriptions)) == len(model_descriptions)

        self._model_descriptions = model_descriptions
        self._global_transformations = global_transformations
        self._models = models
        self._model_transformations = model_transformations
        self._model_hyper_params_object = model_hyper_params_object
        self._model_hyper_params_grid = model_hyper_params_grid
        self._splitter = splitter
        self._resampler_function = resampler_function
        self._results = None
        self._model_persistence_manager = model_persistence_manager
        self._resampler_persistence_manager = resampler_persistence_manager
        self._parallelization_cores = parallelization_cores

    def search(self, data: pd.DataFrame, target_variable: str):
        data_x = data.drop(columns=target_variable)
        data_y = data[target_variable]

        training_indexes, _ = self._splitter.split(target_values=data_y)
        # we don't need the holdout data.. we will send the data to the ModelTrainer which will use the same
        # splitter and will use the training/holdout data appropriately
        # so, we will only get the training data, use that with the tuner, then the tuner's best model, refit
        # the entire training set with the specific model/hyper-params (same training set is used under the
        # hood in the ModelTrainer because we pass the same splitter), and then will evaluate on holdout set
        # which the tuner did not see

        # NOTE: we do NOT need to get the expected columns (like we do for the Resampler and ModelTrainer; for
        # the purposes of ensuring that when we split, uncommon values are missing from being encoded
        # (dummy/one-hot/etc.) and then appearing in the holdout set). The reason we don't need to is because
        # A) the resampler only sees the training set and handles this problem itself and B) the fitter
        # sees ALL the data, but makes the same split on the holdout data, ensuring the holdout data was never
        # used by the Resampler, and also handles the problem itself.)

        # pre-transformed training/holdout sets
        train_data_x_not_transformed = data_x.iloc[training_indexes]
        train_data_y = data_y[training_indexes]
        assert len(train_data_x_not_transformed) == len(train_data_y)

        tuner_results = list()
        holdout_scores = list()
        # for each model: tune; get the best hyper-parameters; train_predict_eval all the training data on the
        # best hyper-parameters; then calculate the final model on the holdout data
        for index in range(len(self._models)):
            local_model_description = self._model_descriptions[index]
            local_model = self._models[index]
            transformations = None
            if self._global_transformations is not None:
                transformations = [x.clone() for x in self._global_transformations]
            if self._model_transformations[index] is not None:
                # if transformations is not None, then it contains global transformations and those need to
                # come before the model transformations
                transformations = self._model_transformations[index] if transformations is None \
                    else transformations + self._model_transformations[index]

            local_model_trans = transformations
            local_model_params_object = self._model_hyper_params_object[index]
            local_model_params_grid = self._model_hyper_params_grid[index]

            local_model_pers_manager = self._model_persistence_manager.clone() if self._model_persistence_manager else None  # noqa
            if local_model_pers_manager is not None:
                # we have a PersistenceManager, we need to ensure each key (e.g. saved file name) is unique
                # because keys across various Tuners are not guaranteed to be unique;
                # keys within Tuners are only unique for each model/hyper-param combination (plus whatever
                # the resampler does), but we might have the same models/hyper-params passed into the
                # searcher; the difference, for example, might be the the transformations
                local_model_pers_manager.set_sub_structure(sub_structure='tune_' + local_model_description)

            local_resampler_pers_manager = self._resampler_persistence_manager.clone() \
                if self._resampler_persistence_manager else None  # noqa
            if local_resampler_pers_manager is not None:
                local_resampler_pers_manager.set_sub_structure(sub_structure='tune_' +
                                                                             local_model_description)

            # clone all the objects to be reused with the ModelTrainer after tuning
            local_resampler_function = self._resampler_function(local_model.clone(),
                                                                None if local_model_trans is None else
                                                                [x.clone() for x in local_model_trans])
            tuner = GridSearchModelTuner(resampler=local_resampler_function,
                                         hyper_param_object=None if local_model_params_object is None
                                                       else local_model_params_object.clone(),
                                         params_grid=local_model_params_grid,
                                         model_persistence_manager=local_model_pers_manager,
                                         resampler_persistence_manager=local_resampler_pers_manager,
                                         parallelization_cores=self._parallelization_cores)

            # noinspection PyProtectedMember
            # before we tune, we need to steel (i.e. clone) the Scores from the resampler so we can use the
            # same ones on the holdout set
            scores = tuner._resampler_factory.get()._score_factory.get()

            tuner.tune(data_x=train_data_x_not_transformed,
                       data_y=train_data_y)
            # GridSearchTunerResults.resampled_stats will have resampled means/st_devs
            tuner_results.append(tuner.results)

            # set prefix rather than sub_structure for refitting model on all data
            local_model_pers_manager = self._model_persistence_manager.clone() if self._model_persistence_manager else None  # noqa
            if local_model_pers_manager is not None:
                # if we have a PersistenceManager, we need to ensure each key (e.g. saved file name) is unique
                # we might have the same models passed into the searcher; the difference, for example, might
                # be the the transformations; but we ensure the model descriptions are unique, so use that
                local_model_pers_manager.set_key_prefix(prefix='final_' + local_model_description + '_')

            # verify that the fitter uses the same training data as the Tuner (i.e. the indexes used for the
            # training data in the fitter match the index used to pass in data to the Tuner)
            def train_callback(transformed_training_data, _1, _2):
                assert all(transformed_training_data.index.values == training_indexes)

            # do not have to clone these objects again, since they won't be reused after this
            fitter = ModelTrainer(model=local_model,
                                  model_transformations=local_model_trans,
                                  splitter=self._splitter,
                                  scores=scores,
                                  persistence_manager=local_model_pers_manager,
                                  train_callback=train_callback)

            if local_model_params_object is not None:
                # if the params object is not None, then we tuned across params and we need to get the best
                # combination
                local_model_params_object.update_dict(tuner.results.best_hyper_params)

            # re-train_predict_eval on entire training set using the best hyper_params.
            # we are passing in all the data, but the Fitter will split the data according to the same
            # Splitter that we used to get the training data to pass into the Tuner.
            fitter.train_predict_eval(data=data,
                                      target_variable=target_variable,
                                      hyper_params=local_model_params_object)

            # get the best model
            holdout_scores.append(fitter.holdout_scores)

        # noinspection PyTypeChecker
        self._results = SearcherResults(model_descriptions=self._model_descriptions,
                                        model_names=[type(x).__name__ for x in self._models],
                                        tuner_results=tuner_results,
                                        holdout_scores=holdout_scores)

    @property
    def results(self) -> SearcherResults:
        if self._results is None:
            raise ModelNotFittedError()

        return self._results
