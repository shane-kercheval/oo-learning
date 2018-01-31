
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


## Example

```python

def whatever():
	pass
```
