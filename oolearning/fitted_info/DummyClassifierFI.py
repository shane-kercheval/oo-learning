from matplotlib import figure

from oolearning.fitted_info.FittedInfoBase import FittedInfoBase


class DummyClassifierFI(FittedInfoBase):

    @property
    def results_summary(self) -> object:
        return None

    @property
    def feature_importance(self) -> dict:
        # noinspection PyTypeChecker
        return None

    @property
    def graph(self) -> figure.Figure:
        # noinspection PyTypeChecker
        return None
