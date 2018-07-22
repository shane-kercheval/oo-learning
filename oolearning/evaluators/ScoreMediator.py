"""
ScoreMediator is needed because different ML problems calculate "scores" with different information.

For example, supervised learning calculates various scores by comparing the actual (target) values with the
    predicted values. Whereas unsupervised learning, such as clustering, defines measures to compare the
    "compactness" and the "separateness" of clusters, which require the original (transformed) dataset and
    the predicted clusters.

A ModelTrainer, for example, doesn't (and shouldn't) know whether it is dealing with a supervised or
    unsupervised problem, and shouldn't know about the differences or the logic to handle those differences.

Instead, the ModelTrainer (and other objects) knows that it has to calculate a score, and will hand off the
    data to on object that does know how to handle it, which will be the ScoreMediator.

The ScoreMediator will simply know which base Score objects (e.g. ScoreActualPredictedBase,
    ScoreClusteringBase) need which information, and will pass the necessarily information accordingly.
    That way, the logic is decoupled from the objects that use Score objects.
"""
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase
from oolearning.evaluators.ScoreClusteringBase import ScoreClusteringBase


class ScoreMediator:
    @staticmethod
    def calculate(score, data_x, actual_target_variables, predicted_values) -> float:
        if isinstance(score, ScoreActualPredictedBase):
            return score.calculate(actual_values=actual_target_variables, predicted_values=predicted_values)
        elif isinstance(score, ScoreClusteringBase):
            return score.calculate(clustered_data=data_x, clusters=predicted_values)
        else:
            raise ValueError('`{}` not known in ScoreMediator'.format(str(type(score))))
