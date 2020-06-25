import pandas as pd
import numpy as np
import pickle
import csv

"""
    Read data from csv file, change all to numerical values, and pickle it
"""
def readData():
    # put all values in csv
    df = pd.read_csv("recipies.csv")

    # delete the column of ids
    del df["Unnamed: 0"]

    # print(df["cuisine"].value_counts())

    # change categorical values to be numeric
    cleanupNums = {"cuisine": {"korean": 1, "indian": 0, "chinese":1, "japanese": 1, "thai": 1}}
    df.replace(cleanupNums, inplace=True)
    
    # print(df)

    df.to_pickle("recipies_cleaned.pkl")


readData()