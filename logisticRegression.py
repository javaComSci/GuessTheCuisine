import pandas as pd
import numpy as np
import pickle

"""
    Model of logistic regression for the data
"""
class LogisticRegression:
    def __init__(self, trainingX, trainingY, testingX, testingY):
        self.trainingX = trainingX
        self.trainingY = trainingY
        self.testingX = testingX
        self.testingY = testingY
    