import numpy as np
import pandas as pd


# noinspection SpellCheckingInspection
class SklearnPredictProbabilityMixin:
    """
    Calls sklearn's model.predict_proba() and converts the results into a dataframe with the corresponding
        class names as columns
    """
    # noinspection PyUnresolvedReferences,PyMethodMayBeStatic
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(model_object.predict_proba(data_x))
        predictions.columns = model_object.classes_
        predictions.index = data_x.index  # ensure the index is the same
        return predictions


class SklearnPredictArrayMixin:
    """
    Calls sklearn's model.predict() function and returns an ndarray
    """
    # noinspection PyUnresolvedReferences,PyMethodMayBeStatic
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> np.ndarray:
        return model_object.predict(data_x)
