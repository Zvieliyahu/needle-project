import pandas as pd
from textblob import TextBlob
from process_helper import *
import nltk
from typing import Dict, List
from nltk.corpus import stopwords
import spacy
from collections import Counter
from cleanData import clean_presidential_speeches

nltk.download('stopwords')

# Load stop words (English)
stop_words = set(stopwords.words('english')) | set(UNRELATED_TOPIC_WORDS)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
for word in UNRELATED_TOPIC_WORDS:
    nlp.vocab[word].is_stop = True

# GLOBALS:
FIRST_THRESHOLD = 0.6
SECOND_THRESHOLD = 0.8


def predict_topics(words_by_topic: Dict) -> str:
    """
    Predicting topics based on number of words per topic.
    :param words_by_topic: dictionary of topic: number of words.
    :return: string of topics (separated by commas)
    """
    # Not classifying if did not find any words
    if not words_by_topic:
        return "None"

    total_count = 0
    for c in words_by_topic.values():
        total_count += c

    # Normalizing count by topics
    normalized_count = {}
    most_common_topic = None
    for topic, count in words_by_topic.items():
        normalized_count[topic] = count / total_count
        # Getting most common topic for prediction
        if most_common_topic is None:
            most_common_topic = topic, normalized_count[topic]
        elif most_common_topic[1] < normalized_count[topic]:
            most_common_topic = topic, normalized_count[topic]

    # If one dominant topic found assigning it
    if most_common_topic[1] > FIRST_THRESHOLD:
        return most_common_topic[0]

    # If no dominant topic found returning most dominant topics:
    topic_coverage = 0
    topics = []
    while topic_coverage < SECOND_THRESHOLD:
        max_topic = max(normalized_count, key=normalized_count.get)
        topics.append(max_topic)
        topic_coverage += normalized_count[max_topic]
        del normalized_count[max_topic]

    # Joining list to be a string separated by commas.
    return ', '.join([f"{word}" for word in topics])


def classify_topic(text, n=30):
    """
    Counting the most common words in a speech after preprocessing the text and assigning topics based on count.
    :param text: Raw text input
    :param n: Number of top words to return
    :return: Formatted string of top topics
    """
    # Process the text with spaCy
    doc = nlp(text.lower())

    # Filter: remove stopwords, punctuation, numbers, and non-alphabetic tokens
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_.lower() not in stop_words
    ]

    # Count word frequencies
    word_counts = Counter(tokens)

    # Get top `n` words
    top_words = word_counts.most_common(n)

    # Converting top words to topics and summing its frequencies by topic
    matched_topics = {}
    for word, count in top_words:
        for topic, keywords in TOPICS_FOR_CLASSIFICATION.items():
            if word in keywords:
                if topic in matched_topics.keys():
                    matched_topics[topic] += count
                else:
                    matched_topics[topic] = count

    # Predicting the subject:
    return predict_topics(matched_topics)


def extract_sentiments(speech: str) -> float:
    """
    Extract a sentiment from a speech, score > 0 is positive, score < 0 is negative.
    :param speech: string of speech
    :return: positivity score
    """
    blob = TextBlob(speech)
    sentiment = blob.sentiment.polarity
    # possibly process more
    return sentiment


if __name__ == "__main__":
    speeches_df = clean_presidential_speeches(r'Data\presidential_speeches.xlsx')
    speeches_df['topics'] = speeches_df['speech'].apply(classify_topic)
    speeches_df.to_excel("speeches_with_topics.xlsx", index=False)