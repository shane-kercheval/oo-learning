from enum import unique, Enum


@unique
class Metric(Enum):
    ACCURACY = 'accuracy'
    AUC_PRECISION_RECALL = 'AUC_PrecisionRecall'
    AUC_ROC = 'AUC_ROC'
    DENSITY_BASED_CLUSTERING_VALIDATION = 'DBCV'
    ERROR_RATE = 'error_rate'
    F1_SCORE = 'F1'
    FBETA_SCORE = 'F_BETA'
    KAPPA = 'kappa'
    MEAN_ABSOLUTE_ERROR = 'MAE'
    MEAN_ABSOLUTE_PERCENTAGE_ERROR = 'MAPE'
    MEAN_SQUARED_ERROR = 'MSE'
    MEAN_SQUARED_PERCENTAGE_ERROR = 'MSPE'
    NEGATIVE_PREDICTIVE_VALUE = 'negative_predictive_value'
    POSITIVE_PREDICTIVE_VALUE = 'positive_predictive_value'
    R_SQUARED = 'r_squared'
    ROOT_MEAN_SQUARE_ERROR = 'RMSE'
    ROOT_MEAN_SQUARE_LOGARITHMIC_ERROR = 'RMSLE'
    SENSITIVITY = 'sensitivity'
    SILHOUETTE = 'silhouette'
    SPECIFICITY = 'specificity'
