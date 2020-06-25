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
        self.theta = [0] * len(trainingX.columns)
    
    
    """
        Fit the parameters of the model given the learning rate
    """
    def fit(self, learningRate):
        print(self.theta)
    