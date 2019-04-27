import os
import shutil
import unittest
from math import isclose

import dill as pickle
import numpy as np

from bayes_opt import BayesianOptimization

from oolearning import *
from tests.MockClassificationModelWrapper import MockClassificationModelWrapper
from tests.MockEvaluator import MockUtilityEvaluator, MockCostEvaluator
from tests.MockHyperParams import MockHyperParams
from tests.MockResampler import MockResampler
from tests.TestHelper import TestHelper
from tests.TimerTestCase import TimerTestCase


# noinspection PyMethodMayBeStatic,SpellCheckingInspection,PyTypeChecker
class BayesianOptimizationTests(TimerTestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_BayesianOptimization(self):
        def black_box_function(x, y):
            """Function with unknown internals we wish to maximize.

            This is just serving as an example, for all intents and
            purposes think of the internals of this function, i.e.: the process
            which generates its output values, as unknown.
            """
            return -x ** 2 - (y - 1) ** 2 + 1

        # Bounded region of parameter space
        pbounds = {'x': (2, 4), 'y': (-3, 3)}

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(
            init_points=2,
            n_iter=3,
        )

        # print(optimizer.max)
        #
        # for i, res in enumerate(optimizer.res):
        #     print("Iteration {}: \n\t{}".format(i, res))
