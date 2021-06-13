from typing import List

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.GridSearchTunerResults import GridSearchTunerResults


class SearcherResults:
    def __init__(self,
                 model_descriptions: List[str],
                 model_names: List[str],
                 tuner_results: List[GridSearchTunerResults],
                 holdout_scores=List[List[ScoreBase]]):
        """
        :param tuner_results: list of GridSearchTunerResults (one per model)
        :param holdout_scores: list of (list of Scores). Each outer list item (one per model),
            contains a list of Scores (same Scores as the Tuner)
        """
        self._model_descriptions = model_descriptions
        self._model_names = model_names
        self._tuner_results = tuner_results
        self._holdout_scores = holdout_scores

    @property
    def tuner_results(self) -> List[GridSearchTunerResults]:
        """
        :return: a list of TunerResult objects
        """
        return self._tuner_results

    @property
    def holdout_score_objects(self) ->List[List[ScoreBase]]:
        """
        :return: List of Lists of Scores. Each item in the outer list corresponds to a specific model/
            model-description.  Each of those items has a list of Scores, corresponding to the
            Scores passed into the Searcher.
        """
        return self._holdout_scores

    @property
    def model_names(self) -> List[str]:
        return self._model_names

    @property
    def model_descriptions(self) -> List[str]:
        return self._model_descriptions

    @property
    def holdout_scores(self) -> pd.DataFrame:
        """
        Score values for the holdout sets
        # TODO: update documentation
        """
        # get all the scores for each model (each model will have the same Evaluators)
        scores = self._holdout_scores[0]
        # get all the columns that the tuner_results will have
        score_columns = [x.name for x in scores]
        holdout_accuracies = [[x.value for x in evaluator] for evaluator in self._holdout_scores]

        return pd.DataFrame(holdout_accuracies, columns=score_columns, index=self._model_descriptions)

    # noinspection PyUnresolvedReferences
    @property
    def best_tuned_results(self):
        """
        :return: a dataframe with each model + best tuned result as a row
        """
        # get all the scores for each model (each model will have the same Evaluators)
        scores = self.holdout_score_objects[0]
        # get all the columns that the tuner_results will have
        score_columns = [(x.name + '_mean', x.name + '_st_dev', x.name + '_cv')
                             for x in scores]
        score_columns = list(sum(score_columns, ()))  # flatten out list
        # get the best resampling results each model that was tuned
        best_model_series = [x.best_model for x in self.tuner_results]

        # since each model will have different tune results, we need to separate and put in its own column
        hyper_params = [x.loc[list(set(x.index.values).difference(score_columns))].to_dict()
                        for x in best_model_series]  # gets the hyper-params for each model (as a dict)

        assert len(self._model_names) == len(hyper_params)
        assert len(self._model_names) == len(best_model_series)

        best_tune_results_df = pd.DataFrame(columns=['model', 'hyper_params'] + score_columns)
        for index in range(len(self._model_names)):
            tune_results_series = best_model_series[index].\
                drop(list(set(best_model_series[index].index.values).difference(score_columns)))
            tune_results_series['hyper_params'] = hyper_params[index]
            tune_results_series['model'] = self._model_names[index]
            best_tune_results_df = best_tune_results_df.\
                append(tune_results_series.loc[['model', 'hyper_params'] + score_columns])

        best_tune_results_df.index = self._model_descriptions

        return best_tune_results_df

    @property
    def best_model_index(self):
        """
        :return: returns the index of the best model based on the holdout accuracies (i.e. based on the first
            Score specified when creating the Resampler passed into the ModelSearcher constructor.
            This index can then be used to index on the `tuner_results` and `holdout_score_objects` properties
        """

        # get the first score in each list (1 list per model)
        score_list = [x[0] for x in self._holdout_scores]
        # should all be the same type of score
        assert all([type(score_list[0]) == type(x) for x in score_list])
        # since Scores implement the __lt__ function based on the better_than function, sorting the
        # Scores gets the indexes of the best sorted
        indexes = np.argsort(score_list)
        return indexes[0]  # return the index of the first i.e. "best" model based on the sorted Scores

# single item is a model
# each model has been tuned so it (the ones with hyper-params at least) have multiple sub-models (i.e. 1 for each hyper-param combination)
# each model has a "best" hyper-params sub-model based on the tuned/resampled results
    # found in `best_tuned_results`, which the the best submodel per model, with associated holdout scores
    # after each "model" has tuned across all sub-model/hyper-param-combos, the best model is chosen, and the model & best hyper-params are refit on all of the training data, and scored on a holdout set
        # these holdout scores are found in `holdout_scores`, these are single values so no mean/standard-dev associated with them
# each sub-model has been resampled, so the "best model" for the sub-model has associated resampled data (i.e. all the cross validation scores for each score)
    #

    def plot_resampled_scores(self,
                              metric: Metric=None,
                              score_name: str=None,
                              x_axis_limits: tuple=(0.0, 1.0),
                              show_one_ste_rule: bool=False):
        """
        for each "best" model, show the resamples via boxplot
        :param metric: the metric (corresponding to the Score object) to display (use this parameter or
            `score_name`
        :param score_name: alternative to the `metric` parameter, you can specify the name of the score to
            retrieve; (the name corresponding to the `name` property of the Score object. While the `metric`
            parameter is a convenience when dealing with built in Scores, `score_name` can be used for custom
            score objects.
        :param x_axis_limits: limits for the x-axis
        :param show_one_ste_rule: show a blue line one standard error below the mean of the best model.
        """
        metric_name = metric.value if score_name is None else score_name
        # build the dataframe that will be used to generate the boxplot; 1 column per "best" model
        resamples = pd.DataFrame()
        for index in range(len(self.model_names)):
            cross_val_scores = self.tuner_results[index].best_model_resampler_object.resampled_scores[metric_name]
            column_name = '{0}: {1}'.format(self.model_names[index],
                                            self.model_descriptions[index])
            resamples[column_name] = pd.Series(data=cross_val_scores)

        # ensure correct number of models (columns in `resamples`, and rows in `resampled_stats`
        assert resamples.shape[1] == len(self.model_names)
        # ensure correct number of resamples (rows in `resamples`, and row in `the underlying cross validation
        # scores (of resampled hyper-param))
        assert resamples.shape[0] == len(self.tuner_results[0].best_model_resampler_object.resampled_scores[metric_name])  # noqa

        # get the means to determine the 'best' hyper-param combo
        resample_means = [resamples[column].mean() for column in resamples.columns.values]
        assert len(resample_means) == resamples.shape[1]

        # get the current score object so we can determine if it is a minimizer or maximizer
        score = [x for x in self.holdout_score_objects[0] if x.name == metric_name]
        assert len(score) == 1  # we should just get the current score
        # if the `better_than` function returns True, 0 is "better than" 1 and we have a minimizer
        # for minimizers, we want to return the min, which is the best value, otherwise, return the max
        minimizer = isinstance(score[0], CostFunctionMixin)
        best = min if minimizer else max
        index_of_best_mean = resample_means.index(best(resample_means))

        resamples.boxplot(vert=False, figsize=(10, 10))
        resample_medians = [resamples[column].median() for column in resamples.columns.values]
        plt.axvline(x=max(resample_medians), color='red', linewidth=1)

        if show_one_ste_rule:
            # using means rather than medians because we are calculating standard error (from the mean)
            resamples_of_best_mean = resamples[resamples.columns.values[index_of_best_mean]].values
            one_standard_error_of_best = resamples_of_best_mean.std() / math.sqrt(len(resamples_of_best_mean))
            # noinspection PyUnresolvedReferences
            one_standard_error_rule = resamples_of_best_mean.mean() - one_standard_error_of_best
            plt.axvline(x=one_standard_error_rule, color='blue', linewidth=1)

        # noinspection PyTypeChecker,PyUnresolvedReferences
        if (resamples <= 1).all().all():
            plt.xlim(x_axis_limits[0], x_axis_limits[1])
            # plt.xticks(np.arange(start=x_axis_limits[0], stop=x_axis_limits[1], step=0.05))
            # ax = plt.gca()
            # ax.set_xticklabels(labels=['0']+['{0:.2f}'.format(x) for x in np.arange(start=x_axis_limits[0] + 0.05,
            #                                                                         stop=x_axis_limits[1] + 0.05, step=0.05)]+['1'],  # noqa
            #                    rotation=20,
            #                    ha='right')
        plt.title('{0} ({1})'.format('Resampling Scores Per `Best` Models',
                                     metric.name if score_name is None else score_name),
                  loc='right')
        plt.tight_layout()
        plt.gca().get_yticklabels()[index_of_best_mean].set_color('red')
        plt.gca().invert_yaxis()

    def plot_holdout_scores(self):
        """
        NOTE: only shows the "tuned" hyper-params i.e. hyper-params that were tuned over >1 values.
        :return:
        """
        scores = self.holdout_score_objects[0]
        # if the Score is a Cost Function it is a 'minimizer'
        minimizers = [isinstance(x, CostFunctionMixin) for x in scores]

        score_columns = [x.name for x in scores]
        score_values = self.holdout_scores.loc[:, score_columns]

        num_rows = len(score_values)
        num_cols = len(score_values.columns)
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(num_cols):
            truths = [True] * num_cols
            truths[i] = False
            mask = np.array(num_rows * [truths], dtype=bool)
            color_values = np.ma.masked_where(mask, score_values)
            # "_r" value after color means invert colors (small values are darker)
            ax.pcolormesh(color_values, cmap='Blues_r' if minimizers[i] else 'Greens')

        for y in range(score_values.shape[0]):
            for x in range(score_values.shape[1]):
                plt.text(x + .5, y + .5, '%.3f' % score_values.iloc[y, x],
                         horizontalalignment='center',
                         verticalalignment='center')

        ax.set_xticks(np.arange(start=0.5, stop=len(score_columns), step=1))
        ax.set_xticklabels(score_columns, rotation=35, ha='right')

        labels = self.holdout_scores.index.values

        y_tick_positions = np.arange(start=0, stop=len(labels)) + 0.5
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        plt.tight_layout()
