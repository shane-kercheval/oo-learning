import pandas as pd
from matplotlib import figure

from oolearning.fitted_info.FittedInfoBase import FittedInfoBase


class SvmFI(FittedInfoBase):
    @property
    def results_summary(self) -> pd.DataFrame:
        pass

    @property
    def feature_importance(self) -> dict:
        pass

    @property
    def graph(self) -> figure.Figure:
        pass
