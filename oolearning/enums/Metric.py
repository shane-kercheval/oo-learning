from enum import unique, Enum


@unique
class Metric(Enum):
    ROOT_MEAN_SQUARE_ERROR = 'RMSE'
    MEAN_ABSOLUTE_ERROR = 'MAE'
    MEAN_SQUARED_ERROR = 'MSE'
    ACCURACY = 'accuracy'
    AUC_ROC = 'AUC_ROC'
    AUC_PRECISION_RECALL = 'AUC_PrecisionRecall'
    KAPPA = 'kappa'
    F1_SCORE = 'F1'
    SPECIFICITY = 'specificity'
    SENSITIVITY = 'sensitivity'
    POSITIVE_PREDICTIVE_VALUE = 'positive_predictive_value'
    NEGATIVE_PREDICTIVE_VALUE = 'negative_predictive_value'
    ERROR_RATE = 'error_rate'
