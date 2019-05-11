from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic,SpellCheckingInspection,PyTypeChecker
class BayesianOptimizationTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_BayesianOptimization_Classification_UtilityFunction(self):

        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns='Survived')

        holdout_y = data.iloc[holdout_indexes][target_variable]
        holdout_x = data.iloc[holdout_indexes].drop(columns='Survived')

        score_list = [AucRocScore(positive_class=1)]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        bounds_lgb = {
            'max_depth': (-1, 10),
            'num_leaves': (10, 100)
        }

        init_points = 1
        n_iter = 4

        model_tuner = BayesianOptimizationModelTuner(resampler=resampler,
                                                     hyper_param_object=LightGBMHP(match_type=True),
                                                     parameter_bounds=bounds_lgb,
                                                     init_points=init_points,
                                                     n_iter=n_iter,
                                                     verbose=0,
                                                     seed=6
                                                     )
        model_tuner.tune(data_x=training_x, data_y=training_y)

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianOptimization_tuner_results.txt')

        assert model_tuner.results.number_of_cycles == init_points + n_iter

        best_indexes = model_tuner.results.resampled_stats.sort_values(by='AUC_ROC_mean',
                                                                       ascending=False).index.values
        assert all(model_tuner.results.sorted_best_indexes == best_indexes)
        assert all(model_tuner.results.sorted_best_indexes == [2, 3, 4, 0, 1])
        assert model_tuner.results.best_index == best_indexes[0]
        assert model_tuner.results.best_model['AUC_ROC_mean'] == 0.8729365330743747
        assert model_tuner.results.best_hyper_params == {'max_depth': -1, 'num_leaves': 10}

        # noinspection PyUnresolvedReferences
        assert all(model_tuner.results.optimizer_results['AUC_ROC'].values ==
                   model_tuner.results.resampled_stats['AUC_ROC_mean'].values)

        # noinspection PyUnresolvedReferences
        TestHelper.save_df(model_tuner.results.optimizer_results.round(8),
                           'data/test_Tuners/test_BayesianOptimization_tuner_optimizer_results.txt')

        # for each row in the underlying results dataframe, make sure the hyper-parameter values in the row
        # correspond to the same hyper_param values found in the Resampler object
        for index in range(len(model_tuner.results._tune_results_objects)):
            assert model_tuner.results._tune_results_objects.iloc[index].max_depth == model_tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['max_depth']  # noqa
            assert model_tuner.results._tune_results_objects.iloc[index].num_leaves == model_tuner.results._tune_results_objects.iloc[index].resampler_object.hyper_params.params_dict['num_leaves']  # noqa

        # Params found by Bayesian Optimization
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        pipeline = TransformerPipeline(transformations=transformations)
        transformed_training_data = pipeline.fit_transform(training_x)
        transformed_holdout_data = pipeline.transform(holdout_x)

        model = LightGBMClassifier()
        hyper_params = LightGBMHP()
        hyper_params.update_dict(model_tuner.results.best_hyper_params)
        model.train(data_x=transformed_training_data, data_y=training_y,
                    hyper_params=hyper_params)
        tuned_holdout_predictions = model.predict(data_x=transformed_holdout_data)
        score_value = AucRocScore(positive_class=1).calculate(actual_values=holdout_y,
                                                              predicted_values=tuned_holdout_predictions)
        assert score_value == 0.7928853754940711

    def test_BayesianOptimization_Regression_CostFunction(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]

        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns=target_variable)

        holdout_y = data.iloc[holdout_indexes][target_variable]
        holdout_x = data.iloc[holdout_indexes].drop(columns=target_variable)

        score_list = [RmseScore()]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        bounds_lgb = {
            'alpha': (0.001, 2),
            'l1_ratio': (0, 1)
        }

        init_points = 1
        n_iter = 4

        model_tuner = BayesianOptimizationModelTuner(resampler=resampler,
                                                     hyper_param_object=ElasticNetRegressorHP(),
                                                     parameter_bounds=bounds_lgb,
                                                     init_points=init_points,
                                                     n_iter=n_iter,
                                                     verbose=0,
                                                     seed=6
                                                     )
        model_tuner.tune(data_x=training_x, data_y=training_y)

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianOptimization_Regression_CostFunction_results.txt')  # noqa

        assert model_tuner.results.number_of_cycles == init_points + n_iter

        # best indexes have the lowest value
        best_indexes = model_tuner.results.resampled_stats.sort_values(by='RMSE_mean',
                                                                       ascending=True).index.values
        assert all(model_tuner.results.sorted_best_indexes == best_indexes)
        assert all(model_tuner.results.sorted_best_indexes == [2, 0, 3, 1, 4])
        assert model_tuner.results.best_index == best_indexes[0]

        # noinspection PyUnresolvedReferences
        assert all(model_tuner.results.optimizer_results['RMSE'].values * -1 ==
                   model_tuner.results.resampled_stats['RMSE_mean'].values)

        # noinspection PyUnresolvedReferences
        TestHelper.save_df(model_tuner.results.optimizer_results.round(8),
                           'data/test_Tuners/test_BayesianOptimization_Regression_CostFunction_optimizer_results.txt')  # noqa

        # for each row in the underlying results dataframe, make sure the hyper-parameter values in the row
        # correspond to the same hyper_param values found in the Resampler object
        for index in range(len(model_tuner.results._tune_results_objects)):
            assert model_tuner.results._tune_results_objects.iloc[index].alpha == \
                   model_tuner.results._tune_results_objects.iloc[
                       index].resampler_object.hyper_params.params_dict['alpha']  # noqa
            assert model_tuner.results._tune_results_objects.iloc[index].l1_ratio == \
                   model_tuner.results._tune_results_objects.iloc[
                       index].resampler_object.hyper_params.params_dict['l1_ratio']  # noqa

    def test_BayesianHyperOptModelTuner(self):
        import numpy as np
        from hyperopt import hp, tpe, fmin

        # Single line bayesian optimization of polynomial function
        best = fmin(fn=lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x),
                    space=hp.normal('x', 4.9, 0.5), algo=tpe.suggest,
                    max_evals=2000)

        temp = lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x)
        x = list(range(1, 10))
        y = temp(x)

        import pandas as pd
        pd.DataFrame({'x':x, 'y':y}).plot()

