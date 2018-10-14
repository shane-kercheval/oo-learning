import pandas as pd
import numpy as np
import itertools

from oolearning.transformers.TransformerBase import TransformerBase


class EncodeInteractionEffectsTransformer(TransformerBase):
    """
    Takes the columns and encodes all the combinations of the values from each column.
    Returns the dataframe with all of the original columns (minus the columns being encoded) plus the
    column combinations.

    :param columns: the columns to encode
    :possible_values: the possible values that might be found in the dataset. Used to ensure that if the
        dataset doesn't contain all the possible values, the necessary columns still get created so that
        future data still works. `possible_values` should be a dictionary, each key should refer to a column

    **Either** `columns` should be supplied or `possible_values` should be supplied, but not both (and not
        neither)
    """
    def __init__(self,
                 columns: list=None,
                 possible_values: dict=None):
        super().__init__()

        # should only have one parameter or the other
        assert (columns is not None or possible_values is not None) and \
               not (columns is not None and possible_values is not None)

        assert columns is None or isinstance(columns, list)
        assert possible_values is None or isinstance(possible_values, dict)

        self._columns = columns
        self._possible_values = possible_values

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        if self._columns is not None:
            # if _possible_values is null then we use the columns and figure out the possible combinations
            # based on the values that exist in the dataset. If future data contains new values, then this
            # will explode because it will refer to a dummy column that doesn't exist
            columns = self._columns
            unique_values = {column: list(np.sort(data_x[column].unique())) for column in self._columns}

        else:
            columns = list(self._possible_values.keys())
            unique_values = self._possible_values

        unique_value_list = [y if isinstance(y, list) else [y] for x, y in unique_values.items()]
        value_combinations = list(itertools.product(*unique_value_list))  # creates a set per combination

        # new_column_temp creates a list of each column/values per combination
        new_column_temp = [[columns[y] + str(x[y]) for y in range(len(x))] for x in value_combinations]
        expected_column_names = ['_'.join(map(str, x)) for x in new_column_temp]
        expected_column_names.sort()

        # return state
        return {'expected_column_names': expected_column_names,
                'columns': columns}

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        columns = self._state['columns']
        expected_column_names = self._state['expected_column_names']

        # create values based on the column name and value
        values = pd.Series(['_'.join([column + str(data_x.iloc[index][column]) for column in columns])
                            for index in range(len(data_x))])
        dummied_data = pd.get_dummies(data=values,
                                      columns=None,
                                      drop_first=False,
                                      sparse=False)

        # make sure that there aren't any NEW columns i.e. new values in the dataset
        # this will cause unexpected issues in models
        assert len(set(dummied_data.columns.values).difference(expected_column_names)) == 0

        dummied_data.index = data_x.index

        dummied_data.sort_index(axis=1, inplace=True)

        return pd.concat([data_x[[x for x in data_x.columns.values if x not in columns]],
                          dummied_data.reindex(columns=expected_column_names, fill_value=0)], axis=1)

    def peak(self, data_x: pd.DataFrame):
        """
        rather than peaking use possible_values, it's cleaner and a better pattern
        """
        pass
