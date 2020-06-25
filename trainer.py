import pandas as pd
import numpy as np
import pickle


"""
    Split the data into training and testing sets
"""
def splitData():
    df = pd.read_pickle("recipies_cleaned.pkl")


if __name__ == "__main__":
    trainingX, trainingY, testingX, testingY = splitData()
    model = LogisticRegression(trainingX, trainingY, testingX, testingY)