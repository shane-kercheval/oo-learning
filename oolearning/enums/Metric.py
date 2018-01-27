from enum import unique, Enum


@unique
class Metric(Enum):
    ROOT_MEAN_SQUARE_ERROR = 'RMSE'
    MEAN_ABSOLUTE_ERROR = 'MAE'
    MEAN_SQUARED_ERROR = 'MSE'
    AREA_UNDER_CURVE = 'AUC'
    KAPPA = 'kappa'
    SPECIFICITY = 'specificity'
    SENSITIVITY = 'sensitivity'
