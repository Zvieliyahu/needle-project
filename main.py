import pandas as pd
from cleanData import *
from processData import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    cleaned_presidential_speeches = clean_presidential_speeches('Data\presidential_speeches.xlsx')
    for i in range(100):
        speech = cleaned_presidential_speeches.loc[i, 'speech']
        print(extractSentiments(speech))         # -> ערך חיובי
        print(extractTopic(speech, threshold=0.7))  # -> ['Environment', ...]
