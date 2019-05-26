import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin


class TunerResultsComparison:

    def __init__(self, results_dict: dict):
        """
        :param results_dict: key should be a friendly name i.e. way to identify the tuner results and
            value should be the TunerResultsBase
        """
        self._results_dict = results_dict

    def boxplot(self,
                title="Resample Results of Best Tuned Model",
                figure_size=(8, 5)):
        results_list = []
        for result_name, result_object in self._results_dict.items():
            df = result_object.best_model_resampler_object.resampled_scores.copy()
            df['model'] = result_name
            results_list.append(df)

        results = pd.concat(results_list, sort=True)
        columns = [x for x in results.columns.values if x != 'model']
        results_melt = results.melt(id_vars='model', value_vars=columns, var_name='Scores')

        main_score_is_cost_function = isinstance(
            list(self._results_dict.values())[0].best_model_resampler_object.scores[0][0], CostFunctionMixin)
        main_score = list(self._results_dict.values())[0].best_model_resampler_object.score_names[0]
        ascending = True if main_score_is_cost_function else False  # min value first if cost function

        fig, ax = plt.subplots(figsize=figure_size)
        sns.boxplot(data=results_melt,
                    hue='Scores',
                    y='model',
                    x='value',
                    order=results.groupby('model')[main_score].mean().sort_values(
                        ascending=ascending).index.values,  # noqa
                    orient='h',
                    ax=ax
                    )
        plt.title(title)  # You can change the title here
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.subplots_adjust(left=0.25)
        plt.tight_layout()
