import pandas as pd
import numpy as np
import pickle
import logisticRegression


"""
    Split the data into training and testing sets
"""
def splitData():
    # read the pickled file
    df = pd.read_pickle("recipies_cleaned.pkl")

    # randomly shuffle the rows
    df = df.sample(frac=1)

    # say that 80% of the samples are for training and rest is for testing
    trainingSize = int(0.8 * len(df))
    # print(trainingSize, len(df))

    # split into training and testing sets
    trainingX = df.loc[:, df.columns != "cuisine"][:trainingSize]
    trainingY = df.iloc[:,0][:trainingSize]
    testingX = df.loc[:, df.columns != "cuisine"][trainingSize + 1:]
    testingY = df.iloc[:,0][trainingSize + 1:]
    # print(len(testingY), testingY)

    return trainingX, trainingY, testingX, testingY



if __name__ == "__main__":
    trainingX, trainingY, testingX, testingY = splitData()
    model = logisticRegression.LogisticRegression(trainingX, trainingY, testingX, testingY)
    model.fitWithCrossValidation(0.01)