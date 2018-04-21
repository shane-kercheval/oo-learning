
**THIS PROJECT IS NOT READY TO BE CONSUMED.**






# oo-learning

OO-Learning is a simple Python machine learning library based on Object Oriented design principles.

The goal of the project is to allow users to quickly explore data and search for top machine learning algorithm *candidates* for a given dataset.

More specifically, this library implements common workflows when exploring data and trying various featuring engineering techniques and machine learning algorithms.

The power of object-oriented design means the user can easily interchange various objects (e.g. transformations, evaluators, resampling techniques), and even extend or build their own.

After model selection, if implementing the model in a production system, the user may or may not want to use a more mature library such as [scikit-learn](https://github.com/scikit-learn/scikit-learn).


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


- `utility` function measures how **good** the model is
- `cost` function measure how **bad** the model is


# Examples

## Classification Dataset

https://github.com/shane-kercheval/oo-learning/tree/master/examples/classification-titanic

* [Exploring a dataset](https://github.com/shane-kercheval/oo-learning/blob/master/examples/classification-titanic/1-%20Exploring%20the%20Titanic%20Dataset.ipynb)
* TBD 

## Regression Dataset

* TBD 

# Available Models


## Regression

- LinearRegressor
- LassoRegressor
- RidgeRegressor
- ElasticNetRegressor
- RandomForestRegressor

#### TBD

- Partial Least Squares
- Neural Networks
- Support Vector Machines
- MARS / FDA
- k-nearest neighbors
- trees
- rules
- bagged trees
- random forest
- boosted trees
- Cubist




## Two-Class Classification

- DummyClassifier
- LogisticClassifier
- RandomForestClassifier


#### TBD

- Partial Least Squares
- Neural Networks
- Support Vector Machines
- MARS / FDA
- k-nearest neighbors
- trees
- rules
- bagged trees
- random forest
- boosted trees
- {LQRM}DA
- Nearest Shrunken Centroids
- Naive Bayes
- C5.0
- Stacker/Blenders


## Multi-Class Classification

- SoftmaxLogisticClassifier
- RandomForestClassifier

#### TBD

- One vs All
- One vs One

# Unit Tests

The unit tests in this project are all found in the tests directory.

In the terminal, run the following in the root project directory:

> python -m unittest discover ./tests
