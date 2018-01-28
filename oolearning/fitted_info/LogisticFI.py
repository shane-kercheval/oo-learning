import pandas as pd
from matplotlib import figure

from oolearning.fitted_info.FittedInfoBase import FittedInfoBase


class LogisticFI(FittedInfoBase):
    @property
    def results_summary(self) -> pd.DataFrame:
        #return self._model_object.summary()
        pass

    @property
    def summary_stats(self) -> dict:
        pass

    @property
    def warnings(self):
        pass

    @property
    def feature_importance(self) -> dict:
        pass

    @property
    def graph(self) -> figure.Figure:
        pass
