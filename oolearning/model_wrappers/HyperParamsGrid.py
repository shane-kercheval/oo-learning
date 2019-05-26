import pandas as pd
import itertools


class HyperParamsGrid:
    def __init__(self, params_dict):
        self._params_dict = params_dict

    @property
    def params_grid(self) -> pd.DataFrame:
        params_list = [y if isinstance(y, list) else [y] for x, y in self._params_dict.items()]
        grid_df = pd.DataFrame(list(itertools.product(*params_list)))
        grid_df.columns = self._params_dict.keys()

        return grid_df

    @property
    def param_names(self) -> list:
        return [key for key in self._params_dict]

    @property
    def tuned_hyper_params(self) -> list:
        """
        The user can specify hyper_parameters in the `params_dict` that are being 'specified' rather than
        'tuned'. In other words, for the hyper-param being 'specified', there is just a single value, as
        opposed to a list of values to 'tune'. In this case, we may not care about seeing this value in
        various graphs/etc., and we only want the hyper-params that are being tuned over multiple values,
        which is what 'tuned_hyper_params' gives.

        For example, if we initialized HyperParamsGrid with a `params_dict` of

            {'criterion': 'gini',
             'max_features': [3, 6, 11],
             'n_estimators': [10, 100, 500],
             'min_samples_leaf': [1, 50, 100]}

        then `tuned_hyper_params` would return `max_features`, `n_estimators`, and `min_samples_leaf`, because
        each of those hyper-params are being tuned across multiple values, where as `criterion` is just being
        'specified' rather than 'tuned'.

        The `hyper_params` property would return all four keys from the above dictionary.
        """
        return [key for key, value in self._params_dict.items() if isinstance(value, list) and len(value) > 1]
