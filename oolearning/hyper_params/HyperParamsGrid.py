import pandas as pd
import itertools


class HyperParamsGrid:
    def __init__(self, params_dict):
        self._params_dict = params_dict

    @property
    def params_grid(self):
        params_list = [y if isinstance(y, list) else [y] for x, y in self._params_dict.items()]
        grid_df = pd.DataFrame(list(itertools.product(*params_list)))
        grid_df.columns = self._params_dict.keys()

        return grid_df
