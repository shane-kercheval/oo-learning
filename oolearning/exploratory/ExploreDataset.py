from oolearning.exploratory.ExploreDatasetBase import ExploreDatasetBase


class ExploreDataset(ExploreDatasetBase):

    def plot_against_target(self, feature: str):
        raise NotImplementedError('No target feature')
