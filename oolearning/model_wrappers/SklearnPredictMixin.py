import numpy as np
import pandas as pd


class SklearnPredictClassifierMixin:
    # noinspection PyMethodMayBeStatic,PyUnresolvedReferences
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> pd.DataFrame:
        predictions = pd.DataFrame(model_object.predict_proba(data_x))
        predictions.columns = model_object.classes_
        return predictions


class SklearnPredictRegressionMixin:
    # noinspection PyMethodMayBeStatic,PyUnresolvedReferences
    def _predict(self, model_object: object, data_x: pd.DataFrame) -> np.ndarray:
        return model_object.predict(data_x)
