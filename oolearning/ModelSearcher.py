from typing import List, Callable

import numpy as np
import pandas as pd

from oolearning.ModelInfo import ModelInfo
from oolearning.SearcherResults import SearcherResults
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.model_processors.ModelFitter import ModelFitter
from oolearning.model_processors.ModelTuner import ModelTuner
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.splitters.DataSplitterBase import DataSplitterBase
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class ModelSearcher:
    def __init__(self,
                 # applied to all data (document fit_transform() on training and transform() on test
                 global_transformations: List[TransformerBase],
                 model_infos: List[ModelInfo],
                 splitter: DataSplitterBase,
                 resampler_function: Callable[[ModelWrapperBase, List[TransformerBase]], ResamplerBase],
                 persistence_manager: PersistenceManagerBase = None):
        """
        # TODO document
        :param persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
            NOTE: a unique key is built up by prefixing the key with the model description
        # TODO: document: we can't define the model and model transformations, everything else we can
        # TODO: document why the lamdba, basically, we can't instantiate a Resampler at this point, and to get around this would probably cause more confusion

        # global_transformations are transformations you want to apply to all the data, regardless of the
        # type of model. For example, regardless of the model, we want to remove the PassengerId and Name
        # fields. These just don't make sense to use as predictors. On the other hand, perhaps for Logistic
        # Regression we also want to Center/Scale the data, but for RandomForest we don't. In those cases,
        # we would want to model specific transformations.

        """
        model_descriptions = [x.description for x in model_infos]
        models = [x.model_wrapper for x in model_infos]
        model_transformations = [x.transformations for x in model_infos]
        model_hyper_params_object = [x.hyper_params for x in model_infos]
        model_hyper_params_grid = [x.hyper_params_grid for x in model_infos]

        # ensure all descriptions are unique
        # length of unique values should be the same as the length of all the values
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
        self._persistence_manager = persistence_manager

    def search(self, data_x: pd.DataFrame, data_y: np.ndarray):
        # split the data into training and holdout data
        # for each model, run a ModelTuner using the resampler_function, creating copies of shit when
        # necessary store each of the TuningResults for each model. Take the best hyper params (if the model
        # has hyper-params), and retrain the model on the entire training set (retrain on entire training set
        # even if no hyper-params), and then get the holdout set value. We will want to track the
        # mean/st_dev resampling value vs the holdout value.
        training_indexes, holdout_indexes = self._splitter.split(target_values=data_y)

        # pre-transformed training/holdout sets
        train_data_x_not_transformed = data_x.iloc[training_indexes]
        train_data_y = data_y[training_indexes]
        assert len(train_data_x_not_transformed) == len(train_data_y)

        holdout_data_x_not_transformed = data_x.iloc[holdout_indexes]
        holdout_data_y = data_y[holdout_indexes]
        assert len(holdout_data_x_not_transformed) == len(holdout_data_y)

        # we want to fit the transformations only on the training data and transform accordingly.
        # Then, when we make predictions, we will transform the "unseen" data based based on the
        # transformations fitted on the training set
        # this is the best way to simulate never-before-seen data with the holdout set.
        pipeline = TransformerPipeline(transformations=self._global_transformations)
        train_data_x = pipeline.fit_transform(data_x=train_data_x_not_transformed)
        holdout_data_x = pipeline.transform(data_x=holdout_data_x_not_transformed)

        assert len(train_data_x) == len(train_data_y)
        assert len(holdout_data_x) == len(holdout_data_y)

        tuner_results = list()
        holdout_evaluators = list()
        # for each model: tune; get the best hyper-parameters; fit all the training data on the best
        # hyper-parameters; then evaluate the final model on the holdout data
        for index in range(len(self._models)):
            local_model_description = self._model_descriptions[index]
            local_model = self._models[index]
            local_model_trans = self._model_transformations[index]
            local_model_params_object = self._model_hyper_params_object[index]
            local_model_params_grid = self._model_hyper_params_grid[index]

            if self._persistence_manager is not None:
                # we have a PersistenceManager, we need to ensure each key (e.g. saved file name) is unique
                # because keys across various Tuners are not guaranteed to be unique;
                # keys within Tuners are only unique for each model/hyper-param combination (plus whatever
                # the resampler does), but we might have the same models/hyper-params passed into the
                # searcher; the difference, for example, might be the the transformations
                self._persistence_manager.set_key_prefix(prefix='tune_' + local_model_description+'_')

            # clone all the objects to be reused with the ModelFitter after tuning
            tuner = ModelTuner(resampler=self._resampler_function(local_model.clone(),
                                                                  None if local_model_trans is None else
                                                                  [x.clone() for x in local_model_trans]),
                               hyper_param_object=None if local_model_params_object is None else local_model_params_object.clone(),  # noqa
                               persistence_manager=self._persistence_manager)

            # noinspection PyProtectedMember
            # before we tune, we need to steel i.e. clone the Evaluators from the resampler so we can use the
            # same ones on the holdout set
            evaluators = [x.clone() for x in tuner._resampler._evaluators]

            tuner.tune(data_x=train_data_x,
                       data_y=train_data_y,
                       params_grid=local_model_params_grid)
            tuner_results.append(tuner.results)  # TunerResults.tune_results will have resampled means/st_devs

            if self._persistence_manager is not None:
                # if we have a PersistenceManager, we need to ensure each key (e.g. saved file name) is unique
                # we might have the same models passed into the searcher; the difference, for example, might
                # be the the transformations; but we ensure the model descriptions are unique, so use that
                self._persistence_manager.set_key_prefix(prefix='holdout_' + local_model_description+'_')

            # do not have to clone these objects again, since they won't be reused after this
            fitter = ModelFitter(model=local_model,
                                 model_transformations=local_model_trans,
                                 evaluators=evaluators,
                                 persistence_manager=self._persistence_manager)

            if local_model_params_object is not None:
                # if the params object is not None, then we tuned across params and we need to get the best
                # combination
                local_model_params_object.update_dict(tuner.results.best_hyper_params)

            # re-fit on entire training set using the best hyper_params.
            fitter.fit(data_x=train_data_x,
                       data_y=train_data_y,
                       hyper_params=local_model_params_object)

            # holdout_evaluators=None ensures that the same holdout_evaluators that were passed into the the
            # ModelFitter constructor will be used.
            fitter.evaluate_holdout(holdout_x=holdout_data_x, holdout_y=holdout_data_y, evaluators=None)

            # get the best model
            holdout_evaluators.append(fitter.holdout_evaluators)

        self._results = SearcherResults(model_descriptions=self._model_descriptions,
                                        model_names=[type(x).__name__ for x in self._models],
                                        tuner_results=tuner_results,
                                        holdout_evaluators=holdout_evaluators)

    @property
    def results(self) -> SearcherResults:
        if self._results is None:
            raise ModelNotFittedError()

        return self._results
