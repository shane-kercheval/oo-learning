from typing import List, Callable, Union

import pandas as pd

from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ModelFitter import ModelFitter
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_processors.ModelTuner import ModelTuner
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.SearcherResults import SearcherResults
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.splitters.DataSplitterBase import DataSplitterBase
from oolearning.transformers.TransformerBase import TransformerBase


class ModelSearcher:
    def __init__(self,
                 # applied to all data (document fit_transform() on training and transform() on test
                 model_infos: List[ModelInfo],
                 splitter: DataSplitterBase,
                 resampler_function: Callable[[ModelWrapperBase, List[TransformerBase]], ResamplerBase],
                 global_transformations: Union[List[TransformerBase], None] = None,
                 resampler_decorators: List[DecoratorBase] = None,
                 persistence_manager: PersistenceManagerBase = None):
        """
        # TODO document
        # TODO document: resampler_decordators is a list of decorators, the decorates are cloned for each
        model i.e. each model uses the same list of decorators
        :param persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
            NOTE: a unique key is built up by prefixing the key with the model description
        # TODO: document: we can't define the model and model transformations, everything else we can
        # TODO: document why the lamdba, basically, we can't instantiate a Resampler at this point, and to get around this would probably cause more confusion

        # TODO: document, the persistence_manger is cloned for each model, and when tuning, the substructure is set to "tune_[model description]" and when fitting on the entire dataset, the prefix is set to "final_[model description]"

        # global_transformations are transformations you want to apply to all the data, regardless of the
        # type of model. For example, regardless of the model, we want to remove the PassengerId and Name
        # fields. These just don't make sense to use as predictors. On the other hand, perhaps for Logistic
        # Regression we also want to Center/Scale the data, but for RandomForestClassifier we don't. In those
        # cases, we would want to model specific transformations.
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
        self._resampler_decorators = resampler_decorators
        self._persistence_manager = persistence_manager

    def search(self, data: pd.DataFrame, target_variable: str):
        # split the data into training and holdout data
        # for each model, run a ModelTuner using the resampler_function, creating copies of shit when
        # necessary store each of the TuningResults for each model. Take the best hyper params (if the model
        # has hyper-params), and retrain the model on the entire training set (retrain on entire training set
        # even if no hyper-params), and then get the holdout set value. We will want to track the
        # mean/st_dev resampling value vs the holdout value.

        # we don't need the holdout data.. we will send the data to the ModelFitter which will use the same
        # splitter and will use the training/holdout data appropriately
        # so, we will only get the training data, use that with the tuner, then the tuner's best model, refit
        # the entire training set with the specific model/hyper-params (same training set is used under the
        # hood in the ModelFitter because we pass the same splitter), and then will evaluate on holdout set
        # which the tuner did not see

        # NOTE: we do NOT need to get the expected columns (like we do for the Resampler and ModelFitter; for
        # the purposes of ensuring that when we split, uncommon values are missing from being encoded
        # (dummy/one-hot/etc.) and then appearing in the holdout set). The reason we don't need to is because
        # A) the resampler only sees the training set and handles this problem itself and B) the fitter
        # sees ALL the data, but makes the same split on the holdout data, ensuring the holdout data was never
        # used by the Resampler, and also handles the problem itself.)
        data_x = data.drop(columns=target_variable)
        data_y = data[target_variable]

        training_indexes, _ = self._splitter.split(target_values=data_y)

        # pre-transformed training/holdout sets
        train_data_x_not_transformed = data_x.iloc[training_indexes]
        train_data_y = data_y[training_indexes]
        assert len(train_data_x_not_transformed) == len(train_data_y)

        tuner_results = list()
        holdout_scores = list()
        # for each model: tune; get the best hyper-parameters; fit all the training data on the best
        # hyper-parameters; then calculate the final model on the holdout data
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

            local_persistence_manager = self._persistence_manager.clone() if self._persistence_manager else None  # noqa
            if local_persistence_manager is not None:
                # we have a PersistenceManager, we need to ensure each key (e.g. saved file name) is unique
                # because keys across various Tuners are not guaranteed to be unique;
                # keys within Tuners are only unique for each model/hyper-param combination (plus whatever
                # the resampler does), but we might have the same models/hyper-params passed into the
                # searcher; the difference, for example, might be the the transformations
                local_persistence_manager.set_sub_structure(sub_structure='tune_' + local_model_description)

            # clone all the objects to be reused with the ModelFitter after tuning
            tuner = ModelTuner(resampler=self._resampler_function(local_model.clone(),
# we want to fit the transformations only on the training data and transform accordingly.
# Then, when we make predictions, we will transform the "unseen" data based based on the
# transformations fitted on the training set
# this is the best way to simulate never-before-seen data with the holdout set.
# we will do this approach both for the resampler and
#TODO document or make note to self that the resampler will transform the data according to each fold (e.g. fit/transform on training folds and transform on holdout fold)
                                                                  None if local_model_trans is None else
                                                                  [x.clone() for x in local_model_trans]),
                               hyper_param_object=None if local_model_params_object is None else local_model_params_object.clone(),  # noqa
                               resampler_decorators=self._resampler_decorators,
                               persistence_manager=local_persistence_manager)

            # noinspection PyProtectedMember
            # before we tune, we need to steel i.e. clone the Scores from the resampler so we can use the
            # same ones on the holdout set
            scores = [x.clone() for x in tuner._resampler._scores]

            tuner.tune(data_x=train_data_x_not_transformed,
                       data_y=train_data_y,
                       params_grid=local_model_params_grid)
            tuner_results.append(tuner.results)  # TunerResults.tune_results will have resampled means/st_devs

            # set prefix rather than sub_structure for refitting model on all data
            local_persistence_manager = self._persistence_manager.clone() if self._persistence_manager else None  # noqa
            if local_persistence_manager is not None:
                # if we have a PersistenceManager, we need to ensure each key (e.g. saved file name) is unique
                # we might have the same models passed into the searcher; the difference, for example, might
                # be the the transformations; but we ensure the model descriptions are unique, so use that
                local_persistence_manager.set_key_prefix(prefix='final_' + local_model_description + '_')

            # verify that the fitter uses the same training data as the Tuner (i.e. the indexes used for the
            # training data in the fitter match the index used to pass in data to the Tuner)
            def train_callback(transformed_training_data, _1, _2):
                assert all(transformed_training_data.index.values == training_indexes)

            # do not have to clone these objects again, since they won't be reused after this
            fitter = ModelFitter(model=local_model,
# TODO document or make note to self that the resampler will transform the data according to each fold (e.g. fit/transform on training folds and transform on holdout fold)
                                 model_transformations=local_model_trans,
# TODO note (and verify to self) that splitter will split in the same way as above, so we don't really need the holdout data above, Fitter will train on the same dataset that the resampler used, and predict on the holdout set, which the resampler did not see
                                 splitter=self._splitter,
                                 scores=scores,
                                 persistence_manager=local_persistence_manager,
                                 train_callback=train_callback)

            if local_model_params_object is not None:
                # if the params object is not None, then we tuned across params and we need to get the best
                # combination
                local_model_params_object.update_dict(tuner.results.best_hyper_params)

            # re-fit on entire training set using the best hyper_params.
            # we are passing in all the data, but the Fitter will split the data according to the same
            # Splitter that we used to get the training data to pass into the Tuner.
            fitter.fit(data=data,
                       target_variable=target_variable,
                       hyper_params=local_model_params_object)

            # get the best model
            holdout_scores.append(fitter.holdout_scores)

        self._results = SearcherResults(model_descriptions=self._model_descriptions,
                                        model_names=[type(x).__name__ for x in self._models],
                                        tuner_results=tuner_results,
                                        holdout_scores=holdout_scores)

    @property
    def results(self) -> SearcherResults:
        if self._results is None:
            raise ModelNotFittedError()

        return self._results
