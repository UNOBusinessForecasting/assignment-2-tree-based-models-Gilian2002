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

def checkNumbers(series):
    for i in series:
        if not isinstance(i, (float, int, np.int64, np.float64)):
            return False
    return True


class testCases(unittest.TestCase):
    def testValidPred(self):
        self.assertTrue((len(list(pred))==1000 and checkNumbers(pred)), "Make sure your prediction consists of integers\nor floating point numbers, and is a list or array of 744\nfuture predictions!")
        
