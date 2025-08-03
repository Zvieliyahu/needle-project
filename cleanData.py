import pandas as pd


def clean_presidential_speeches(path: str):
    """
    Cleaning the presidential_speech.csv file.
    :param path: path of the file
    :return: cleaned file
    """
    presidential_speeches = pd.read_excel(path)
    # remove vice president column
    presidential_speeches = presidential_speeches.drop('Vice President', axis=1)
    # remove rows with  None
    presidential_speeches.dropna(inplace=True)
    # make speech and info lower case
    presidential_speeches['speech'] = presidential_speeches['speech'].str.lower()
    presidential_speeches['info'] = presidential_speeches['info'].str.lower()
    presidential_speeches['date'] = pd.to_datetime(presidential_speeches['date'], errors='coerce')
    # Removing audience reactions
    presidential_speeches['speech'] = presidential_speeches['speech'].str.replace(r'\((.*?)\)', '', regex=True)
    return presidential_speeches
