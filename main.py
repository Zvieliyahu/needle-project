import pandas as pd

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    presidential_speeches = pd.read_excel('presidential_speeches.xlsx')
    presidential_speeches = presidential_speeches.drop('Vice President', axis=1)
    presidential_speeches.dropna(inplace=True)
    cleaned_presidential_speeches =

