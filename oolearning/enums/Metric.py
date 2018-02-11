from enum import unique, Enum


@unique
class Metric(Enum):
    ROOT_MEAN_SQUARE_ERROR = 'RMSE'
    MEAN_ABSOLUTE_ERROR = 'MAE'
    MEAN_SQUARED_ERROR = 'MSE'
    ACCURACY = 'accuracy'
    AREA_UNDER_CURVE = 'AUC'
    KAPPA = 'kappa'
    F1_SCORE = 'F1'
    SPECIFICITY = 'specificity'
    SENSITIVITY = 'sensitivity'
    ERROR_RATE = 'ErrorRate'
