from matplotlib import figure

from oolearning.fitted_info.FittedInfoBase import FittedInfoBase


# TODO: NOT FINISHED / TESTED, ONLY COMPLETED ENOUGH TO TEST MODEL TUNER
class RandomForestFI(FittedInfoBase):
    @property
    def results_summary(self) -> object:
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
