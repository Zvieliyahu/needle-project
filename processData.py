import pandas as pd
from textblob import TextBlob

# Extract sentiment per speech
def extractSentiments(speech : str) -> float:
    blob = TextBlob(speech)
    sentiment = blob.sentiment.polarity
    # possibly process more
    return sentiment

