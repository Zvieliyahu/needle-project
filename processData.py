import pandas as pd
from textblob import TextBlob
from transformers import pipeline

labels = [
    "Economy", "Foreign Policy", "National Security", "Civil Rights", "Healthcare",
    "Education", "Environment", "Law and Order", "Government and Institutions",
    "Technology and Innovation", "Religion and Morality", "Campaign and Politics"
]
# Extract sentiment per speech
def extractSentiments(speech : str) -> float:
    blob = TextBlob(speech)
    sentiment = blob.sentiment.polarity
    # possibly process more
    return sentiment

def extractTopic(speech:str,threshold)->list:
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    if not isinstance(speech, str) or speech.strip() == "":
        return []
    result = classifier(speech, candidate_labels=labels, multi_label=True)
    tagged = [label for label, score in zip(result["labels"], result["scores"]) if score > threshold]
    return tagged