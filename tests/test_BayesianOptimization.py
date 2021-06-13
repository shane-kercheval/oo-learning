import numpy as np

from oolearning import *
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


class ModelDecorator(DecoratorBase):
    def __init__(self):
        self._model_list = list()

    def decorate(self, **kwargs):
        self._model_list.append(kwargs['model'])


class TransformerDecorator(DecoratorBase):
    def __init__(self):
        self._pipeline_list = list()

    def decorate(self, **kwargs):
        self._pipeline_list.append(kwargs['transformer_pipeline'])


# noinspection PyMethodMayBeStatic,SpellCheckingInspection,PyTypeChecker,PyUnresolvedReferences
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
        training_x, training_y, holdout_x, holdout_y = splitter.split_sets(data=data,
                                                                           target_variable=target_variable)
        score_list = [AucRocScore(positive_class=1)]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=-1)  # adds parallelization (per repeat)

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
        assert all(model_tuner.results.sorted_best_indexes == [3, 0, 2, 4, 1])
        assert model_tuner.results.best_index == best_indexes[0]
        assert model_tuner.results.best_model['AUC_ROC_mean'] == 0.8693921743812189
        assert model_tuner.results.best_hyper_params == {'max_depth': 5, 'num_leaves': 38}

        # noinspection PyUnresolvedReferences
        assert all(model_tuner.results.optimizer_results['AUC_ROC'].values ==
                   model_tuner.results.resampled_stats['AUC_ROC_mean'].values)

        TestHelper.check_plot('data/test_Tuners/test_BayesianOptimization_Classification_UtilityFunction__optimizerplot_iteration_mean_scores.png',  # noqa
                              lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianOptimization_Classification_UtilityFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianOptimization_Classification_UtilityFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.AUC_ROC))

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
        assert score_value == 0.7867588932806324

    def test_BayesianOptimization_Regression_CostFunction(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]

        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_x, training_y, _, _ = splitter.split_sets(data=data, target_variable=target_variable)

        score_list = [RmseScore()]

        # define & coonfigure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=-1)  # adds parallelization (per repeat)

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

        TestHelper.check_plot('data/test_Tuners/test_BayesianOptimization_Regression_CostFunction__optimizerplot_iteration_mean_scores.png',  # noqa
                              lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianOptimization_Regression_CostFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianOptimization_Regression_CostFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.ROOT_MEAN_SQUARE_ERROR))  # noqa

        assert model_tuner.results.number_of_cycles == init_points + n_iter

        # best indexes have the lowest value
        best_indexes = model_tuner.results.resampled_stats.sort_values(by='RMSE_mean',
                                                                       ascending=True).index.values
        assert all(model_tuner.results.sorted_best_indexes == best_indexes)
        assert all(model_tuner.results.sorted_best_indexes == [0, 3, 1, 2, 4])
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

    # noinspection PyUnresolvedReferences
    def test_BayesianHyperOptModelTuner_Classification_UtilityFunction(self):
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_x, training_y, holdout_x, holdout_y = splitter.split_sets(data=data,
                                                                           target_variable=target_variable)

        score_list = [AucRocScore(positive_class=1)]

        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)  # adds parallelization (per repeat)

        # space_lgb = {
        #     'max_depth': (-1, 10),
        #     'num_leaves': (10, 100)
        # }
        from hyperopt import hp
        space_lgb = {
            'max_depth': hp.choice('max_depth', range(-1, 10)),
            'num_leaves': hp.choice('num_leaves', range(10, 100)),
        }

        max_evaluations = 5

        model_tuner = BayesianHyperOptModelTuner(resampler=resampler,
                                                 hyper_param_object=LightGBMHP(),
                                                 space=space_lgb,
                                                 max_evaluations=max_evaluations,
                                                 seed=6)
        model_tuner.tune(data_x=training_x, data_y=training_y)

        assert model_tuner.results.best_index == 4
        assert model_tuner.results.transformation_names == []
        assert model_tuner.results.model_hyper_param_names == ['max_depth', 'num_leaves']
        assert model_tuner.results.hyper_param_names == model_tuner.results.model_hyper_param_names

        expected_best_params = {'max_depth': 7, 'num_leaves': 13}
        assert model_tuner.results.best_hyper_params == expected_best_params
        assert model_tuner.results.best_hyper_params == model_tuner.results.best_model_hyper_params
        assert model_tuner.results.best_model_hyper_params == expected_best_params
        assert model_tuner.results.best_transformations == []

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        assert max_evaluations == model_tuner.results.number_of_cycles

        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)

        for index in range(len(model_tuner.resampler_decorators)):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params,
                                                   mapping={'min_gain_to_split': 'min_split_gain',
                                                            'min_sum_hessian_in_leaf': 'min_child_weight',
                                                            'min_data_in_leaf': 'min_child_samples',
                                                            'bagging_fraction': 'subsample',
                                                            'bagging_freq': 'subsample_freq',
                                                            'feature_fraction': 'colsample_bytree',
                                                            'lambda_l1': 'reg_alpha',
                                                            'lambda_l2': 'reg_lambda'})

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 4 for original transformations + 1 StatelessTransformer
                assert len(local_transformations) == 5
                assert isinstance(local_transformations[0], RemoveColumnsTransformer)
                assert isinstance(local_transformations[1], CategoricConverterTransformer)
                assert isinstance(local_transformations[2], ImputationTransformer)
                assert isinstance(local_transformations[3], DummyEncodeTransformer)
                assert isinstance(local_transformations[4], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Classification_UtilityFunction_results.txt')  # noqa

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Classification_UtilityFunction__optimizerplot_iteration_mean_scores.png',  # noqa
                               lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Classification_UtilityFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Classification_UtilityFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.AUC_ROC))

        assert model_tuner.results.hyper_param_names == list(space_lgb.keys())
        assert model_tuner.results.resampled_stats.shape[0] == max_evaluations

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['AUC_ROC_mean'].idxmax(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [AucRocScore(positive_class=1)]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = LightGBMHP()
        hp.update_dict(model_tuner.results.best_hyper_params)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert OOLearningHelpers.dict_is_subset(subset=model_tuner.results.best_hyper_params,
                                                superset=resampler.results.hyper_params.params_dict)
        assert resampler.results.score_means['AUC_ROC'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['AUC_ROC_mean']

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
        holdout_predictions = model.predict(data_x=transformed_holdout_data)
        score_value = AucRocScore(positive_class=1).calculate(actual_values=holdout_y,
                                                              predicted_values=holdout_predictions)
        assert score_value == 0.7907114624505929

    # noinspection PyUnresolvedReferences
    def test_BayesianHyperOptModelTuner_Regression_CostFunction(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]

        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_x, training_y, holdout_x, holdout_y = splitter.split_sets(data=data,
                                                                           target_variable=target_variable)

        score_list = [RmseScore()]
        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)  # adds parallelization (per repeat)

        from hyperopt import hp
        space_lgb = {
            'alpha': hp.uniform('alpha', 0.001, 2),
            'l1_ratio': hp.uniform('l1_ratio', 0, 1),
        }

        max_evaluations = 5
        model_tuner = BayesianHyperOptModelTuner(resampler=resampler,
                                                 hyper_param_object=ElasticNetRegressorHP(),
                                                 space=space_lgb,
                                                 max_evaluations=max_evaluations,
                                                 seed=6)
        model_tuner.tune(data_x=training_x, data_y=training_y)
        assert model_tuner.results.best_index == 0
        assert model_tuner.results.transformation_names == []
        assert model_tuner.results.model_hyper_param_names == ['alpha', 'l1_ratio']
        assert model_tuner.results.hyper_param_names == model_tuner.results.model_hyper_param_names

        expected_best_params = {'alpha': 1.0311739683159624, 'l1_ratio': 0.38070855784252866}
        assert model_tuner.results.best_hyper_params == expected_best_params
        assert model_tuner.results.best_hyper_params == model_tuner.results.best_model_hyper_params
        assert model_tuner.results.best_model_hyper_params == expected_best_params
        assert model_tuner.results.best_transformations == []

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        assert max_evaluations == model_tuner.results.number_of_cycles

        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)

        for index in range(len(model_tuner.resampler_decorators)):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params)

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 1 for original transformations + 1 StatelessTransformer
                assert len(local_transformations) == 2
                assert isinstance(local_transformations[0], RemoveColumnsTransformer)  # added by Resampler
                assert isinstance(local_transformations[1], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Regression_CostFunction_results.txt')  # noqa

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Regression_CostFunction__plot_iteration_mean_scores.png',  # noqa
                               lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Regression_CostFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Regression_CostFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.ROOT_MEAN_SQUARE_ERROR))  # noqa

        assert model_tuner.results.hyper_param_names == list(space_lgb.keys())
        assert model_tuner.results.resampled_stats.shape[0] == max_evaluations

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['RMSE_mean'].idxmin(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]
        score_list = [RmseScore()]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = ElasticNetRegressorHP()
        hp.update_dict(model_tuner.results.best_hyper_params)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert OOLearningHelpers.dict_is_subset(subset=model_tuner.results.best_hyper_params,
                                                superset=resampler.results.hyper_params.params_dict)
        assert resampler.results.score_means['RMSE'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['RMSE_mean']

        # Params found by Bayesian Optimization
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]
        pipeline = TransformerPipeline(transformations=transformations)
        transformed_training_data = pipeline.fit_transform(training_x)
        transformed_holdout_data = pipeline.transform(holdout_x)

        model = ElasticNetRegressor()
        hyper_params = ElasticNetRegressorHP()
        hyper_params.update_dict(model_tuner.results.best_hyper_params)
        model.train(data_x=transformed_training_data, data_y=training_y,
                    hyper_params=hyper_params)
        holdout_predictions = model.predict(data_x=transformed_holdout_data)
        score_value = RmseScore().calculate(actual_values=holdout_y, predicted_values=holdout_predictions)
        assert score_value == 10.001116248623699

    def test_BayesianHyperOptModelTuner_Transformations_Classification_UtilityFunction(self):
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_x, training_y, holdout_x, holdout_y = splitter.split_sets(data=data,
                                                                           target_variable=target_variable)

        score_list = [AucRocScore(positive_class=1)]

        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)  # adds parallelization (per repeat)

        from hyperopt import hp
        space_lgb = {
            # passing None will get removed, but work around is to pass a Transformer that doesn't do anything
            # i.e. EmptyTransformer
            'CenterScale vs Normalize': hp.choice('CenterScale vs Normalize vs None',
                                                  [EmptyTransformer(),
                                                   CenterScaleTransformer(),
                                                   NormalizationTransformer()]),
            'PCA': hp.choice('PCA',
                                                  [EmptyTransformer(),
                                                   PCATransformer()]),
            'max_depth': hp.choice('max_depth', range(-1, 10)),
            'num_leaves': hp.choice('num_leaves', range(10, 100)),
        }

        max_evaluations = 5

        model_tuner = BayesianHyperOptModelTuner(resampler=resampler,
                                                 hyper_param_object=LightGBMHP(),
                                                 space=space_lgb,
                                                 max_evaluations=max_evaluations,
                                                 seed=6)
        model_tuner.tune(data_x=training_x, data_y=training_y)

        assert model_tuner.results.best_hyper_params == {'CenterScale vs Normalize': 'CenterScaleTransformer',
                                                         'PCA': 'EmptyTransformer', 'max_depth': 5,
                                                         'num_leaves': 88}
        assert model_tuner.results.best_index == 1
        assert all(model_tuner.results.sorted_best_indexes == [1, 3, 0, 2, 4])
        assert model_tuner.results.number_of_cycles == max_evaluations
        assert model_tuner.results.hyper_param_names == ['CenterScale vs Normalize', 'PCA', 'max_depth',
                                                         'num_leaves']
        assert model_tuner.results.transformation_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.model_hyper_param_names == ['max_depth', 'num_leaves']
        assert model_tuner.results.best_model_hyper_params == {'max_depth': 5, 'num_leaves': 88}

        assert len(model_tuner.results.best_transformations) == 2
        assert isinstance(model_tuner.results.best_transformations[0], CenterScaleTransformer)
        assert isinstance(model_tuner.results.best_transformations[1], EmptyTransformer)
        assert not model_tuner.results.best_transformations[0].has_executed
        assert not model_tuner.results.best_transformations[1].has_executed

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        assert max_evaluations == model_tuner.results.number_of_cycles

        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles

        for index in range(len(model_tuner.resampler_decorators)):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[
                index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params,
                                                   mapping={'min_gain_to_split': 'min_split_gain',
                                                            'min_sum_hessian_in_leaf': 'min_child_weight',
                                                            'min_data_in_leaf': 'min_child_samples',
                                                            'bagging_fraction': 'subsample',
                                                            'bagging_freq': 'subsample_freq',
                                                            'feature_fraction': 'colsample_bytree',
                                                            'lambda_l1': 'reg_alpha',
                                                            'lambda_l2': 'reg_lambda'})

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 4 for original transformations + 2 hyper-paramed + 1 StatelessTransformer
                assert len(local_transformations) == 7
                assert isinstance(local_transformations[0], RemoveColumnsTransformer)
                assert isinstance(local_transformations[1], CategoricConverterTransformer)
                assert isinstance(local_transformations[2], ImputationTransformer)
                assert isinstance(local_transformations[3], DummyEncodeTransformer)

                # local_transformations has the transformations that were actually used in the Resampler
                # via the decorators
                # index_transformation_objects contains the transformations that were tuned
                index_transformation_objects = list(
                    model_tuner.results._transformations_objects[index].values())  # noqa
                assert type(index_transformation_objects[0]) is type(local_transformations[4])  # noqa
                assert type(index_transformation_objects[1]) is type(local_transformations[5])  # noqa

                assert isinstance(local_transformations[6], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Classification_UtilityFunction_results.txt')  # noqa

        TestHelper.check_plot(
            'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Classification_UtilityFunction__optimizerplot_iteration_mean_scores.png',  # noqa
            lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Classification_UtilityFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Classification_UtilityFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.AUC_ROC))

        assert model_tuner.results.hyper_param_names == list(space_lgb.keys())
        assert model_tuner.results.resampled_stats.shape[0] == max_evaluations

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['AUC_ROC_mean'].idxmax(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [AucRocScore(positive_class=1)]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = LightGBMHP()
        hp.update_dict(model_tuner.results.best_model_hyper_params)
        resampler.append_transformations(model_tuner.results.best_transformations)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert OOLearningHelpers.dict_is_subset(subset=model_tuner.results.best_model_hyper_params,
                                                superset=resampler.results.hyper_params.params_dict)
        assert resampler.results.score_means['AUC_ROC'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['AUC_ROC_mean']

        # Params found by Bayesian Optimization
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]
        pipeline = TransformerPipeline(transformations=transformations +
                                                       model_tuner.results.best_transformations)
        transformed_training_data = pipeline.fit_transform(training_x)
        transformed_holdout_data = pipeline.transform(holdout_x)

        model = LightGBMClassifier()
        hyper_params = LightGBMHP()
        hyper_params.update_dict(model_tuner.results.best_model_hyper_params)
        model.train(data_x=transformed_training_data, data_y=training_y,
                    hyper_params=hyper_params)
        holdout_predictions = model.predict(data_x=transformed_holdout_data)
        TestHelper.save_string(model,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Classification_UtilityFunction_final_model.txt')  # noqa

        assert OOLearningHelpers.dict_is_subset(subset=model_tuner.results.best_model_hyper_params,
                                                superset=model.model_object.get_params())

        score_value = AucRocScore(positive_class=1).calculate(actual_values=holdout_y,
                                                              predicted_values=holdout_predictions)

        assert score_value == 0.7867588932806324

    def test_BayesianHyperOptModelTuner_Transformations_Regression_CostFunction(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]

        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_x, training_y, holdout_x, holdout_y = splitter.split_sets(data=data,
                                                                           target_variable=target_variable)

        score_list = [RmseScore()]
        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)  # adds parallelization (per repeat)

        from hyperopt import hp
        space_lgb = {
            # passing None will get removed, but work around is to pass a Transformer that doesn't do anything
            # i.e. EmptyTransformer
            'CenterScale vs Normalize': hp.choice('CenterScale vs Normalize vs None',
                                                  [EmptyTransformer(),
                                                   CenterScaleTransformer(),
                                                   NormalizationTransformer()]),
            'PCA': hp.choice('PCA',
                             [EmptyTransformer(),
                              PCATransformer()]),
            'alpha': hp.uniform('alpha', 0.001, 2),
            'l1_ratio': hp.uniform('l1_ratio', 0, 1),
        }
        max_evaluations = 5
        model_tuner = BayesianHyperOptModelTuner(resampler=resampler,
                                                 hyper_param_object=ElasticNetRegressorHP(),
                                                 space=space_lgb,
                                                 max_evaluations=max_evaluations,
                                                 seed=6)
        model_tuner.tune(data_x=training_x, data_y=training_y)

        expected_best_transformations = {'CenterScale vs Normalize': 'EmptyTransformer',
                                         'PCA': 'EmptyTransformer'}
        expected_best_params = {'alpha': 1.6544370627160525, 'l1_ratio': 0.5364332695946842}

        assert model_tuner.results.best_hyper_params == {**expected_best_transformations,
                                                         **expected_best_params}
        assert model_tuner.results.best_index == 1
        assert all(model_tuner.results.sorted_best_indexes == [1, 0, 2, 3, 4])
        assert model_tuner.results.number_of_cycles == max_evaluations
        assert model_tuner.results.hyper_param_names == ['CenterScale vs Normalize', 'PCA', 'alpha', 'l1_ratio']  # noqa
        assert model_tuner.results.transformation_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.model_hyper_param_names == list(expected_best_params.keys())
        assert model_tuner.results.best_model_hyper_params == expected_best_params

        assert len(model_tuner.results.best_transformations) == 2
        assert isinstance(model_tuner.results.best_transformations[0], EmptyTransformer)
        assert isinstance(model_tuner.results.best_transformations[1], EmptyTransformer)
        assert not model_tuner.results.best_transformations[0].has_executed
        assert not model_tuner.results.best_transformations[1].has_executed

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        assert max_evaluations == model_tuner.results.number_of_cycles
        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)

        for index in range(len(model_tuner.resampler_decorators)):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params)

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 1 for original transformations + 2 hyper-params + 1 StatelessTransformer
                assert len(local_transformations) == 4
                assert isinstance(local_transformations[0], RemoveColumnsTransformer)  # added by Resampler

                # local_transformations has the transformations that were actually used in the Resampler
                # via the decorators
                # index_transformation_objects contains the transformations that were tuned
                index_transformation_objects = list(model_tuner.results._transformations_objects[index].values())  # noqa
                assert type(index_transformation_objects[0]) is type(local_transformations[1])  # noqa
                assert type(index_transformation_objects[1]) is type(local_transformations[2])  # noqa

                assert isinstance(local_transformations[3], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Regression_CostFunction_results.txt')  # noqa

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Regression_CostFunction__plot_iteration_mean_scores.png',  # noqa
                               lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Regression_CostFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Regression_CostFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.ROOT_MEAN_SQUARE_ERROR))  # noqa

        assert model_tuner.results.hyper_param_names == list(space_lgb.keys())
        assert model_tuner.results.resampled_stats.shape[0] == max_evaluations

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['RMSE_mean'].idxmin(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]
        score_list = [RmseScore()]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = ElasticNetRegressorHP()
        hp.update_dict(model_tuner.results.best_model_hyper_params)
        resampler.append_transformations(model_tuner.results.best_transformations)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert OOLearningHelpers.dict_is_subset(subset=model_tuner.results.best_model_hyper_params,
                                                superset=resampler.results.hyper_params.params_dict)
        assert resampler.results.score_means['RMSE'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['RMSE_mean']

        # Params found by Bayesian Optimization
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]
        pipeline = TransformerPipeline(transformations=transformations +
                                                       model_tuner.results.best_transformations)
        transformed_training_data = pipeline.fit_transform(training_x)
        transformed_holdout_data = pipeline.transform(holdout_x)

        model = ElasticNetRegressor()
        hyper_params = ElasticNetRegressorHP()
        hyper_params.update_dict(model_tuner.results.best_model_hyper_params)
        model.train(data_x=transformed_training_data, data_y=training_y,
                    hyper_params=hyper_params)
        holdout_predictions = model.predict(data_x=transformed_holdout_data)
        TestHelper.save_string(model,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_Regression_CostFunction_final_model_string.txt')  # noqa

        score_value = RmseScore().calculate(actual_values=holdout_y, predicted_values=holdout_predictions)
        assert score_value == 10.000002422021112

    def test_BayesianHyperOptModelTuner_Transformations_only_Classification_UtilityFunction(self):
        data = TestHelper.get_titanic_data()
        target_variable = 'Survived'
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)
        training_x, training_y, _, _ = splitter.split_sets(data=data, target_variable=target_variable)

        score_list = [AucRocScore(positive_class=1)]

        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)  # adds parallelization (per repeat)

        from hyperopt import hp
        space_lgb = {
            # passing None will get removed, but work around is to pass a Transformer that doesn't do anything
            # i.e. EmptyTransformer
            'CenterScale vs Normalize': hp.choice('CenterScale vs Normalize vs None',
                                                  [EmptyTransformer(),
                                                   CenterScaleTransformer(),
                                                   NormalizationTransformer()]),
            'PCA': hp.choice('PCA',
                                                  [EmptyTransformer(),
                                                   PCATransformer()]),
        }

        max_evaluations = 5

        model_tuner = BayesianHyperOptModelTuner(resampler=resampler,
                                                 hyper_param_object=LightGBMHP(),
                                                 space=space_lgb,
                                                 max_evaluations=max_evaluations,
                                                 seed=6)
        model_tuner.tune(data_x=training_x, data_y=training_y)

        assert model_tuner.results.best_hyper_params == {'CenterScale vs Normalize': 'EmptyTransformer',
                                                         'PCA': 'EmptyTransformer'}
        assert model_tuner.results.best_index == 1
        assert all(model_tuner.results.sorted_best_indexes == [1, 0, 2, 3, 4])
        assert model_tuner.results.number_of_cycles == max_evaluations
        assert model_tuner.results.hyper_param_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.transformation_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.model_hyper_param_names == []
        assert model_tuner.results.best_model_hyper_params == {}

        assert len(model_tuner.results.best_transformations) == 2
        assert isinstance(model_tuner.results.best_transformations[0], EmptyTransformer)
        assert isinstance(model_tuner.results.best_transformations[1], EmptyTransformer)
        assert not model_tuner.results.best_transformations[0].has_executed
        assert not model_tuner.results.best_transformations[1].has_executed

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert max_evaluations == model_tuner.results.number_of_cycles
        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles

        for index in range(len(model_tuner.resampler_decorators)):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[
                index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params,
                                                   mapping={'min_gain_to_split': 'min_split_gain',
                                                            'min_sum_hessian_in_leaf': 'min_child_weight',
                                                            'min_data_in_leaf': 'min_child_samples',
                                                            'bagging_fraction': 'subsample',
                                                            'bagging_freq': 'subsample_freq',
                                                            'feature_fraction': 'colsample_bytree',
                                                            'lambda_l1': 'reg_alpha',
                                                            'lambda_l2': 'reg_lambda'})

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 4 for original transformations + 2 hyper-paramed + 1 StatelessTransformer
                assert len(local_transformations) == 7
                assert isinstance(local_transformations[0], RemoveColumnsTransformer)
                assert isinstance(local_transformations[1], CategoricConverterTransformer)
                assert isinstance(local_transformations[2], ImputationTransformer)
                assert isinstance(local_transformations[3], DummyEncodeTransformer)

                # local_transformations has the transformations that were actually used in the Resampler
                # via the decorators
                # index_transformation_objects contains the transformations that were tuned
                index_transformation_objects = list(
                    model_tuner.results._transformations_objects[index].values())  # noqa
                assert type(index_transformation_objects[0]) is type(local_transformations[4])  # noqa
                assert type(index_transformation_objects[1]) is type(local_transformations[5])  # noqa

                assert isinstance(local_transformations[6], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Classification_UtilityFunction_results.txt')  # noqa

        TestHelper.check_plot(
            'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Classification_UtilityFunction__optimizerplot_iteration_mean_scores.png',  # noqa
            lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Classification_UtilityFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Classification_UtilityFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.AUC_ROC))

        assert model_tuner.results.hyper_param_names == list(space_lgb.keys())
        assert model_tuner.results.resampled_stats.shape[0] == max_evaluations

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['AUC_ROC_mean'].idxmax(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                           CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                           ImputationTransformer(),
                           DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

        score_list = [AucRocScore(positive_class=1)]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=LightGBMClassifier(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = LightGBMHP()
        hp.update_dict(model_tuner.results.best_model_hyper_params)
        resampler.append_transformations(model_tuner.results.best_transformations)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert OOLearningHelpers.dict_is_subset(subset=model_tuner.results.best_model_hyper_params,
                                                superset=resampler.results.hyper_params.params_dict)
        assert resampler.results.score_means['AUC_ROC'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['AUC_ROC_mean']

    def test_BayesianHyperOptModelTuner_Transformations_only_Regression_CostFunction(self):
        data = TestHelper.get_cement_data()
        target_variable = 'strength'
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]

        splitter = RegressionStratifiedDataSplitter(holdout_ratio=0.2)
        training_x, training_y, _, _ = splitter.split_sets(data=data, target_variable=target_variable)

        score_list = [RmseScore()]
        model_decorator = ModelDecorator()
        transformer_decorator = TransformerDecorator()
        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            fold_decorators=[model_decorator, transformer_decorator],
            parallelization_cores=0)  # adds parallelization (per repeat)

        from hyperopt import hp
        space_lgb = {
            # passing None will get removed, but work around is to pass a Transformer that doesn't do anything
            # i.e. EmptyTransformer
            'CenterScale vs Normalize': hp.choice('CenterScale vs Normalize vs None',
                                                  [EmptyTransformer(),
                                                   CenterScaleTransformer(),
                                                   NormalizationTransformer()]),
            'PCA': hp.choice('PCA',
                             [EmptyTransformer(),
                              PCATransformer()]),
        }
        max_evaluations = 5
        model_tuner = BayesianHyperOptModelTuner(resampler=resampler,
                                                 hyper_param_object=ElasticNetRegressorHP(),
                                                 space=space_lgb,
                                                 max_evaluations=max_evaluations,
                                                 seed=6)
        model_tuner.tune(data_x=training_x, data_y=training_y)

        expected_best_transformations = {'CenterScale vs Normalize': 'EmptyTransformer',
                                         'PCA': 'EmptyTransformer'}

        assert model_tuner.results.best_hyper_params == expected_best_transformations
        assert model_tuner.results.best_index == 1
        assert all(model_tuner.results.sorted_best_indexes == [1, 0, 2, 3, 4])
        assert model_tuner.results.number_of_cycles == max_evaluations
        assert model_tuner.results.hyper_param_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.transformation_names == ['CenterScale vs Normalize', 'PCA']
        assert model_tuner.results.model_hyper_param_names == []
        assert model_tuner.results.best_model_hyper_params == {}

        assert len(model_tuner.results.best_transformations) == 2
        assert isinstance(model_tuner.results.best_transformations[0], EmptyTransformer)
        assert isinstance(model_tuner.results.best_transformations[1], EmptyTransformer)
        assert not model_tuner.results.best_transformations[0].has_executed
        assert not model_tuner.results.best_transformations[1].has_executed

        # should have cloned the decorator each time, so it should not have been used
        assert len(model_decorator._model_list) == 0
        # should be same number of sets/lists of decorators as there are tuning cycles
        assert len(model_tuner.resampler_decorators) == model_tuner.results.number_of_cycles
        assert max_evaluations == model_tuner.results.number_of_cycles
        # 2 decorators object passed in
        assert all(np.array([len(x) for x in model_tuner.resampler_decorators]) == 2)
        for index in range(len(model_tuner.resampler_decorators)):
            # index = 0
            model_list = model_tuner.resampler_decorators[index][0]._model_list
            assert len(model_list) == 5 * 3  # 5 fold 3 repeat
            trained_params = [x.model_object.get_params() for x in model_list]
            # every underlying training params of the resample should be the same (a Resampler object only
            # uses one combination of hyper-params)
            for y in range(len(trained_params)):
                assert trained_params[y] == trained_params[0]

            # check that the hyper params that are passed in match the hyper params that the underlying
            # model actually trains with
            trained_params = trained_params[0]  # already verified that all the list items are the same
            hyper_params = model_tuner.results._tune_results_objects.resampler_object.values[index].hyper_params.params_dict  # noqa
            TestHelper.assert_hyper_params_match_2(subset=hyper_params, superset=trained_params)

            pipeline_list = model_tuner.resampler_decorators[index][1]._pipeline_list
            for pipeline in pipeline_list:
                local_transformations = pipeline.transformations
                # 1 for original transformations + 2 hyper-params + 1 StatelessTransformer
                assert len(local_transformations) == 4
                assert isinstance(local_transformations[0], RemoveColumnsTransformer)  # added by Resampler

                # local_transformations has the transformations that were actually used in the Resampler
                # via the decorators
                # index_transformation_objects contains the transformations that were tuned
                index_transformation_objects = list(
                    model_tuner.results._transformations_objects[index].values())  # noqa
                assert type(index_transformation_objects[0]) is type(local_transformations[1])  # noqa
                assert type(index_transformation_objects[1]) is type(local_transformations[2])  # noqa

                assert isinstance(local_transformations[3], StatelessTransformer)  # added by Resampler
                assert all([transformation.has_executed for transformation in local_transformations])

        TestHelper.save_string(model_tuner.results,
                               'data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Regression_CostFunction_results.txt')  # noqa

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Regression_CostFunction__plot_iteration_mean_scores.png',  # noqa
                               lambda: model_tuner.results.plot_iteration_mean_scores())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Regression_CostFunction__plot_resampled_stats.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_stats())

        TestHelper.check_plot('data/test_Tuners/test_BayesianHyperOptModelTuner_Transformations_only_Regression_CostFunction__plot_resampled_scores.png',  # noqa
                              lambda: model_tuner.results.plot_resampled_scores(metric=Metric.ROOT_MEAN_SQUARE_ERROR))  # noqa

        assert model_tuner.results.hyper_param_names == list(space_lgb.keys())
        assert model_tuner.results.resampled_stats.shape[0] == max_evaluations

        df = model_tuner.results.resampled_stats
        assert model_tuner.results.best_hyper_params == \
               df.loc[df['RMSE_mean'].idxmin(), model_tuner.results.hyper_param_names].to_dict()

        ######################################################################################################
        # Resample with best_params and see if we get the same loss value i.e. RMSE_mean
        ######################################################################################################
        transformations = [RemoveColumnsTransformer(columns=['fineagg'])]
        score_list = [RmseScore()]

        # define & configure the Resampler object
        resampler = RepeatedCrossValidationResampler(
            model=ElasticNetRegressor(),  # we'll use a Random Forest model
            transformations=transformations,
            scores=score_list,
            folds=5,  # 5 folds with 5 repeats
            repeats=3,
            parallelization_cores=0)  # adds parallelization (per repeat)

        hp = ElasticNetRegressorHP()
        hp.update_dict(model_tuner.results.best_model_hyper_params)
        resampler.append_transformations(model_tuner.results.best_transformations)
        resampler.resample(training_x, training_y, hyper_params=hp)

        assert OOLearningHelpers.dict_is_subset(subset=model_tuner.results.best_model_hyper_params,
                                                superset=resampler.results.hyper_params.params_dict)
        assert resampler.results.score_means['RMSE'] == \
               model_tuner.results.resampled_stats.iloc[model_tuner.results.best_index]['RMSE_mean']
