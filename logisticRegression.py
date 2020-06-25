import pandas as pd
import numpy as np
import pickle
import math


"""
    Model of logistic regression for the data
"""
class LogisticRegression:
    def __init__(self, trainingX, trainingY, testingX, testingY):
        self.trainingX = trainingX
        self.trainingY = trainingY
        self.testingX = testingX
        self.testingY = testingY
        self.rows = trainingX.shape[0]
        self.columns = trainingX.shape[1]
        self.theta = np.zeros((self.columns, 1))
        self.bestTheta = np.zeros((self.columns, 1))
    

    

    """
        Reinitalize theta when training a new time
    """
    def reinitTheta(self):
        self.theta = np.zeros((self.columns, 1))



    """
        Fit the parameters given the training set and learning rate
    """
    def fit(self, trainingX, trainingY, learningRate):
        # overall gradient descent: theta = theta - (alpha/m * X * (1/(1+e^-X*theta) - y))
        
        # get the dot product of theta and x
        xTheta = np.dot(trainingX, self.theta)

        # get the reshape the y training examples
        trainingY = trainingY.values.reshape(xTheta.shape[0], 1)

        # make xTheta negative
        negativeXTheta = -1 * xTheta

        # apply the sigmoid function on the dot product of theta and x
        sigmoidVal = 1/(1 + np.exp(negativeXTheta))

        # find the different of predicted and actual value
        sigmoidMinusY = sigmoidVal - trainingY

        
        



    """
        Fit the parameters of the model given the learning rate with the use of cross validation
    """
    def fitWithCrossValidation(self, learningRate):

        # print(self.trainingX)

        # fit by doing leave-one-out cross validation
        for i in range(self.rows):
            trainingRowIndex = [j for j in range(self.rows) if j != i]
            trainingX = self.trainingX.iloc[trainingRowIndex]
            trainingY = self.trainingY.iloc[trainingRowIndex]
            validationX = self.trainingX.iloc[i]
            validationY = self.trainingY.iloc[i]

            self.fit(trainingX, trainingY, learningRate)
            return
            