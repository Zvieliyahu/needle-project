import pandas as pd


def cleanData(path:str):
    presidential_speeches = pd.read_excel(path)
    presidential_speeches = presidential_speeches.drop('Vice President', axis=1)
    presidential_speeches.dropna(inplace=True)
