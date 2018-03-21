from matplotlib import figure

from oolearning.model_wrappers.FittedInfoBase import FittedInfoBase


# noinspection PyTypeChecker
class MockFittedInfo(FittedInfoBase):

    @property
    def results_summary(self) -> object:
        return 'test_summary'

    @property
    def summary_stats(self) -> dict:
        return None

    @property
    def warnings(self):
        return None

    @property
    def feature_importance(self) -> dict:
        return None

    @property
    def graph(self) -> figure.Figure:
        return None
