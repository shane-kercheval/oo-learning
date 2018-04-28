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
- `Searcher`: A Searcher searches across different models and hyper-params, with the goal of finding the "best" ideal model candidates for further tuning and optimization.
- `Decorator`: Intent is to add responsibility objects dynamically. (For example, to piggy-back off of the Resampler and do a calculation or capture data at the end of each fold.)
    
# Examples

https://github.com/shane-kercheval/oo-learning/tree/master/examples/classification-titanic

* Exploring a dataset
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/1-Exploring%20the%20Titanic%20Dataset.ipynb)
	* Regression (TBD)
* Training a model
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/2-Basic%20Modeling.ipynb)
	* Regression (TBD)
* Resampling data
	* [Classification](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/3-Resampling.ipynb)
	* Regression (TBD)
* Tuning Data
	* Classification (TBD)
	* Regression (TBD)
* Searching Models
	* Classification (TBD)
	* Regression (TBD)
* Advanced Topics
	* Model Aggregation and Stacking (TBD)
	* "resampling decorators" 
		* resample the ideal ROC threshold (TBD)
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
evaluator = TwoClassProbabilityEvaluator(converter=
                                         TwoClassThresholdConverter(threshold=0.5,
                                                                    positive_class='lived'))

# give the objects, which encapsulate the behavior of everything involved with training the model, to our ModelTrainer
trainer = ModelTrainer(model=LogisticClassifier(),
                       model_transformations=transformations,
                       splitter=splitter,
                       evaluator=evaluator)
trainer.train(data=data, target_variable='Survived', hyper_params=LogisticClassifierHP())

# access the holdout metrics
trainer.holdout_evaluator.all_quality_metrics
```

*Code Snippet from [Training a model](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/2-Basic%20Modeling.ipynb) notebook.*

### ModelTuner Snippet

```python
TBD
```
### ModelSearcher Snippet

```python
TBD
```

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
