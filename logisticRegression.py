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
        Apply the sigmoid function
    """
    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(x))




    """
        Do gradient descent to fit the parameters
    """
    def gradientDescent(self, trainingX, trainingY, learningRate):
        # overall gradient descent: theta = theta - (alpha/m * X * (1/(1+e^-X*theta) - y))
        
        # calculate x dot theta
        xTheta = np.dot(trainingX, self.theta)

        # apply the sigmoid to get the hypothesis
        predictions = self.sigmoid(xTheta)

        # reshape the training set
        trainingY = trainingY.values.reshape((trainingX.shape[0], 1))

        # derivative
        gradient = np.dot(trainingX.T, predictions - trainingY)

        # average cost
        gradient = gradient/trainingX.shape[0]

        # learning rate include
        gradient = gradient * learningRate

        # udpate theta
        self.theta = self.theta - gradient

        # print(self.theta)
        return max(gradient)



    """
        Calculate the cost with current parameters
    """
    def calculateCost(self, x, y):
         # calculate x dot theta
        xTheta = np.dot(x, self.theta)

        # apply the sigmoid to get the hypothesis
        sigmoidTheta = self.sigmoid(xTheta)

        totalCost = np.dot(-y.T, np.log(sigmoidTheta)) - (np.dot((1 - y).T, np.log(1 - sigmoidTheta)))

        cost = totalCost/x.shape[0]
        print(cost)




    """
        Fit the parameters given the training set and learning rate
    """
    def fit(self, trainingX, trainingY, learningRate):
        # value for convergence
        oldUpdate = 0

        iterations = 0

        while True:
            iterations += 1
            updated = self.gradientDescent(trainingX, trainingY, learningRate)
            # print(oldUpdate, updated)
            if abs(oldUpdate - updated) < 0.0000000005:
                oldUpdate = updated
                break
            else:
                oldUpdate = updated
            self.calculateCost(trainingX, trainingY)
        
        print(iterations)
        
        


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
            