import pandas as pd
from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.enums.CategoricalEncoding import CategoricalEncoding
from oolearning.transformers.TransformerBase import TransformerBase


class DummyEncodeTransformer(TransformerBase):
    """
    Transforms categorical variables to either DUMMY or ONE-HOT encoding
    """

    def __init__(self, encoding: CategoricalEncoding = CategoricalEncoding.DUMMY):
        super().__init__()
        assert isinstance(encoding, CategoricalEncoding)
        assert encoding is not CategoricalEncoding.NONE  # this wouldn't make sense
        self._encoding = encoding
        self._columns_to_reindex = None

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        ######################################################################################################
        # 1) for each categorical variable, save all the categories, to make sure that we transform the data
        # the same way each time, i.e. same dummy columns, even when the dummy value doesn't exist (e.g.
        # predicting on a single instance)
        #
        # 2) if one-hot encoding, leave all columns, if `dummy`, then drop the first column/category for each
        # original variable
        ######################################################################################################
        num, categorical_features = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                          target_variable=None)

        # save the state as columns, so that we can reindex with original columns
        state = {category: sorted(list(data_x[category].dropna().unique()))
                 for category in categorical_features}
        # save the new dummy columns, so that we can reindex consistently in transform
        starting_index = 1 if self._encoding == CategoricalEncoding.DUMMY else 0  # ignore 1st index if dummy

        self._columns_to_reindex = num
        for column in state.keys():
            new_columns = [column + '_' + str(value) for value in state[column]]  # `str` in case of numeric
            self._columns_to_reindex.extend(new_columns[starting_index:])

        return state

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        ######################################################################################################
        # based on the columns/categories found in the `fit` stage,
        # 1) ensure that all the categories being transformed are a SUBSET of what was previously found (i.e.
        # no new columns/values
        # 2) create a new DataFrame with the corresponding dummy columns
        # 4) reindex according to previously defined columns in `fit`
        ######################################################################################################
        _, categorical_features = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                        target_variable=None)
        found_state = {category: sorted(list(data_x[category].dropna().unique()))
                       for category in categorical_features}

        # ensure no new values
        assert len(set(found_state.keys()).symmetric_difference(set(state.keys()))) == 0
        assert all([set(value).issubset(set(state[key])) for key, value in found_state.items()])

        dummied_data = pd.get_dummies(data=data_x,
                                      columns=categorical_features,
                                      prefix_sep='_',
                                      drop_first=False,  # need to do manually
                                      sparse=False)

        # ensure all columns exist (e.g. if transforming single observation, ensure that columns associated
        # with values not found are still created; ensures columns are dropped consistently for dummy encoding
        return dummied_data.reindex(columns=self._columns_to_reindex, fill_value=0)
