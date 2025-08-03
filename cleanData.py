import pandas as pd

def clean_data(path:str):
    presidential_speeches = pd.read_excel(path)
    #remove vice president column
    presidential_speeches = presidential_speeches.drop('Vice President', axis=1)
    #remove rows with  None
    presidential_speeches.dropna(inplace=True)
    #make speech and info lower case
    presidential_speeches['speech'] = presidential_speeches['speech'].str.lower()
    presidential_speeches['info'] = presidential_speeches['info'].str.lower()