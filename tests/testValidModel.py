import unittest
import statsmodels
import patsy as pt
import pandas as pd
import numpy as np
import sklearn
import statsmodels.discrete.discrete_model as dm

# Import your code from parent directory
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from assignment2 import model, modelFit, pred

# Run the checks

class testCases(unittest.TestCase):
    def testValidModel(self):
        modelType = str(type(model))
        for i in ['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier']:
            if i in modelType:
                self.assertTrue(True)
                return None
        self.assertTrue(False)
