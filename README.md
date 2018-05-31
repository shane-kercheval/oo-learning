# oo-learning

**`oo-learning`** is a simple Python machine learning library based on Object Oriented design principles.

The goal of the project is to allow users to quickly explore data and search for top machine learning algorithm *candidates* for a given dataset.

More specifically, this library implements common workflows when exploring data and trying various featuring engineering techniques and machine learning algorithms.

The power of object-oriented design means the user can easily interchange various objects (e.g. transformations, evaluators, resampling techniques), and even extend or build their own.

After model selection, if implementing the model in a production system, the user may or may not want to use a more mature library such as [scikit-learn](https://github.com/scikit-learn/scikit-learn).

# Conventions / Definitions

## Conventions

- `train` for models
- `fit` for data (e.g. transformations)
- `holdout`
	- (e.g. holdout set) over `test` (e.g. test set)
	- `test` is too overloaded and causes confusion in variable/method names
- `features`
	- over `predictors`, `independent variables`, etc.
- `target`
	- over `response`
- `hyperparameter`
	- over `tuning parameter`

## Class Terminology

- `Converter`: A Converter converts a DataFrame containing predictions (i.e. a standard DataFrame returned by `.predict()`, with continuous values (e.g. probabilities) for each class, as columns) into an array of class predictions.
- `Score`: A Score object contains the logic of a single metric for accessing the performance of a model.
	- `utility` function measures how **good** the model is
	- `cost` function measure how **bad** the model is
- `Evaluator`: An Evaluator object takes the predictions of a model, as well as the actual values, and evaluates the model across many metrics.
- `Transformer`:  A transformer is an object that transforms data-sets by first `fitting` an initial data-set, and saving the values necessary to consistently transform future data-sets based on the fitted data-set.
- `Splitter`: A Splitter splits a dataset into training and holdout sets.
- `ModelWrapper`: A ModelWrapper is a class encapsulating a machine learning model/algorithm.
- `Aggregator`: Aggregators combine the predictions of various models into a single prediction.
- `Stacker`: This class implements a simple 'Model Stacker', taking the predictions of base models and feeding the values into a 'stacking' model.
- `Resampler`: A Resampler accesses the accuracy of the model by training the model many times via a particular resampling strategy.
- `Tuner`:  A ModelTuner uses a Resampler for tuning a single model across various hyper-parameters, finding the "best" hyper-parameters supplied as well as related information.
- `Searcher`: A "Searcher" searches across different models and hyper-parameters (or the same models and hyper-parameters with different tranformations, for example) with the goal of finding the "best" or ideal model candidates for further tuning and optimization.
- `Decorator`: Intent is to add responsibility objects dynamically. (For example, to piggy-back off of the Resampler and do a calculation or capture data at the end of each fold.)
    
# Examples

https://github.com/shane-kercheval/oo-learning/tree/master/examples/classification-titanic

* Exploring a dataset
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/1-Exploring%20the%20Titanic%20Dataset.ipynb)
	* [Regression](https://github.com/shane-kercheval/oo-learning/blob/master/examples/regression-insurance/1-Exploring.ipynb)
* Training a model
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/2-Basic%20Modeling.ipynb)
	* [Regression](https://github.com/shane-kercheval/oo-learning/blob/master/examples/regression-insurance/2-Basic%20Modeling.ipynb)
* Resampling data
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/3-Resampling.ipynb)
	* [Regression](https://github.com/shane-kercheval/oo-learning/blob/master/examples/regression-insurance/3-Resampling.ipynb)
* Tuning Data
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/4-Tuning.ipynb)
	* [Regression](https://github.com/shane-kercheval/oo-learning/blob/master/examples/regression-insurance/4-Tuning.ipynb)
* Searching Models
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/5-Searching.ipynb)
	* [Regression](https://github.com/shane-kercheval/oo-learning/blob/master/examples/regression-insurance/5-Searching.ipynb)
* Advanced Topics
	* [Model Aggregation and Stacking](https://github.com/shane-kercheval/oo-learning/blob/master/examples/regression-insurance/Model%20Aggregation%20and%20Stacking.ipynb)
	* "resampling decorators" (see [Classification Resampling](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/3-Resampling.ipynb) example)
		* resample the ideal ROC threshold (see link above)
	* Caching Models via `PersistenceManager` (TBD)
 
### ModelTrainer Snippet

```python
# define how we want to split the training/holding datasets
splitter = ClassificationStratifiedDataSplitter(holdout_ratio=0.2)

# define the transformations
transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                   CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                   ImputationTransformer(),
                   DummyEncodeTransformer(CategoricalEncoding.DUMMY)]

# Define how we want to evaluate (and convert the probabilities DataFrame to predicted classes)
evaluator = TwoClassProbabilityEvaluator(...)

# give the objects, which encapsulate the behavior of everything involved with training the model, to our ModelTrainer
trainer = ModelTrainer(model=LogisticClassifier(),
                       model_transformations=transformations,
                       splitter=splitter,
                       evaluator=evaluator)
trainer.train(data=data, target_variable='Survived', hyper_params=LogisticClassifierHP())

# access holdout metricsscore_names
trainer.holdout_evaluator.all_quality_metrics
```

*Code Snippet from [Training a model](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/2-Basic%20Modeling.ipynb) notebook.*

### ModelTuner Snippet

```python
# define the transformations
transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                   CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                   ImputationTransformer(),
                   DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

# define the scores, which will be used to compare the performance across hyper-param combinations
# the scores need a Converter, which contains the logic necessary to convert the predicted values to a predicted class.
score_list = [AucRocScore(positive_class='lived'),
              SensitivityScore(...),
              SpecificityScore(...),
              ErrorRateScore(...)]

# define/configure the resampler
resampler = RepeatedCrossValidationResampler(model=RandomForestClassifier(),  # using a Random Forest model
                                             transformations=transformations,
                                             scores=score_list,
                                             folds=5,
                                             repeats=5)
# define/configure the ModelTuner
tuner = ModelTuner(resampler=resampler,
                   hyper_param_object=RandomForestHP())  # Hyper-Parameter object specific to RFTBD

# define the parameter values (and, therefore, combinations) we want to try 
params_dict = dict(criterion='gini',
                   max_features=[1, 5, 10],
                   n_estimators=[10, 100, 500],
                   min_samples_leaf=[1, 50, 100])
grid = HyperParamsGrid(params_dict=params_dict)

tuner.tune(data_x=training_x, data_y=training_y, params_grid=grid)
tuner.results.get_heatmap()
```

*Code Snippet from [Tuning](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/4-Tuning.ipynb) notebook.*

### ModelSearcher Snippet

```python
# Logistic Regression Hyper-Param Grid
log_grid = HyperParamsGrid(params_dict=dict(penalty=['l1', 'l2'],
                                            regularization_inverse=[0.001, 0.01, 0.1, 1, 100, 1000]))

# get the expected columns at the time we do the training, based on the transformations 
columns = TransformerPipeline.get_expected_columns(transformations=global_transformations, data=explore.dataset.drop(columns=[target_variable]))
# Random Forest Hyper-Param Grid
rm_grid = HyperParamsGrid(params_dict=dict(criterion='gini',
                                           max_features=[int(round(len(columns) ** (1 / 2.0))),
                                                         int(round(len(columns) / 2)),
                                                         len(columns) - 1],
                                           n_estimators=[10, 100, 500],
                                           min_samples_leaf=[1, 50, 100]))

# define the models and hyper-parameters that we want to search through
infos = [ModelInfo(description='dummy_stratified',
                   model=DummyClassifier(DummyClassifierStrategy.STRATIFIED),
                   transformations=None,
                   hyper_params=None,
                   hyper_params_grid=None),
         ModelInfo(description='dummy_frequent',
                   model=DummyClassifier(DummyClassifierStrategy.MOST_FREQUENT),
                   transformations=None,
                   hyper_params=None,
                   hyper_params_grid=None),
         ModelInfo(description='Logistic Regression',
                   model=LogisticClassifier(),
                   # transformations specific to this model
                   transformations=[CenterScaleTransformer(),
                                    RemoveCorrelationsTransformer()],
                   hyper_params=LogisticClassifierHP(),
                   hyper_params_grid=log_grid),
         ModelInfo(description='Random Forest',
                   model=RandomForestClassifier(),
                   transformations=None,
                   hyper_params=RandomForestHP(),
                   hyper_params_grid=rm_grid)]

# define the transformations that will be applied to ALL models
global_transformations = [RemoveColumnsTransformer(['PassengerId', 'Name', 'Ticket', 'Cabin']),
                          CategoricConverterTransformer(['Pclass', 'SibSp', 'Parch']),
                          ImputationTransformer(),
                          DummyEncodeTransformer(CategoricalEncoding.ONE_HOT)]

# define the Score objects, which will be used to choose the "best" hyper-parameters for a particular model,
# and compare the performance across model/hyper-params, 
score_list = [AucRocScore(positive_class='lived'),
# the SensitivityScore needs a Converter, 
# which contains the logic necessary to convert the predicted values to a predicted class.
              SensitivityScore(converter=TwoClassThresholdConverter(threshold=0.5, positive_class='lived'))]

# create the ModelSearcher object
searcher = ModelSearcher(global_transformations=global_transformations,
                         model_infos=infos,
                         splitter=ClassificationStratifiedDataSplitter(holdout_ratio=0.25),
                         resampler_function=lambda m, mt: RepeatedCrossValidationResampler(
                             model=m,
                             transformations=mt,
                             scores=score_list,
                             folds=5,
                             repeats=3))
searcher.search(data=explore.dataset, target_variable='Survived')
```

*Code Snippet from [Searcher](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/5-Searching.ipynb) notebook.*

# Known Issues

- Parallelization
  - issue with parallelization when used with RandomForestClassifier or RandomForestRegressor
    - seems to be related to https://github.com/scikit-learn/scikit-learn/issues/7346#event-1241008914
  - issue with parallelization when used with `LinearRegressor`
    - need to use `LinearRegressorSK`
  - cannot use parallelization with callbacks (e.g. RepeatedCrossValidationResampler init's `train_callback` parameter because it cannot be pickled i.e. serialized)
  - parallelization used with RepeatedCrossValidationResampler & Decorators that are meant (i.e. the information retained in the Decorator is meant) to persist across repeats will not work.


# Available Models

`R = Regression; 2C = Two-class Classification; MC = Multi-class Classification`

- AdaBoost (R, 2C, MC)
- Cart Decision Tree (R, 2C, MC)
- Elastic Net (R)
- Gradient Boosting (R, 2C, MC)
- Lasso (R, 2C, MC)
- Linear Regression (R, 2C, MC)
- Logistic (2C)
- Random Forest (R, 2C, MC)
- Ridge (R)
- Softmax Logistic (MC)
- Support Vector Machines (R, 2C)

## Future (TBD)

- One-vs-All & One-vs-One (MC)
- Naive Bayes
- Nearest Shrunken Centroids
- KNN
- C5.0
- Cubist
- Partial Least Squares
- MARS / FDA
- Neural Networks


# Unit Tests

The unit tests in this project are all found in the tests directory.

In the terminal, run the following in the root project directory:

> python -m unittest discover ./tests
