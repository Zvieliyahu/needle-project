import pandas as pd
from textblob import TextBlob
from process_helper import *
import nltk
from typing import Dict, List
from nltk.corpus import stopwords
import spacy
from collections import Counter
from cleanData import clean_presidential_speeches
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from tqdm import tqdm
tqdm.pandas()

nltk.download('stopwords')


"""
                      #######################
                      ## Topic Predictions ##
                      #######################
"""

# Load stop words (English)
stop_words = set(stopwords.words('english')) | set(UNRELATED_TOPIC_WORDS)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
for word in UNRELATED_TOPIC_WORDS:
    nlp.vocab[word].is_stop = True

# GLOBALS - Topic Predictions:
# FIRST_THRESHOLD = 0.6
FIRST_THRESHOLD = 0.4
# SECOND_THRESHOLD = 0.8
SECOND_THRESHOLD = 0.7

MIN_WORDS_COUNTED = 5


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

    # Returning None if there is not enough information
    if total_count <= MIN_WORDS_COUNTED:
        return "None"

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


"""
                      ###########################
                      ## Sentiment Predictions ##
                      ###########################
"""

sentiment_pipeline = pipeline("sentiment-analysis")
CHUNK_SIZE = 350


def extract_sentiments(speech: str):
    """
    Extract a sentiment from a speech, with a label "positive" or "negative" and a score
    between 0 and 1 (0 being most negative and 1 most positive)
    :param speech: string of speech
    :return: a dict of label: score
    """
    cleaned = remove_thanking_phrases(speech)

    # Simple word-based chunking approximation
    words = cleaned.split()
    chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

    labels = []
    scores = []

    for chunk in chunks:
        result = sentiment_pipeline(chunk, truncation=True)[0]
        label = result['label']
        score = result['score']

        # Convert label & score to positivity score
        positivity_score = score if label == "POSITIVE" else 1 - score

        labels.append(label)
        scores.append(positivity_score)

    # Majority label
    majority_label = Counter(labels).most_common(1)[0][0]
    avg_score = sum(scores) / len(scores)

    return {
        "label": majority_label,
        "positivity_score": round(avg_score, 4)
    }


"""
                      #########################
                      ## Emotion Predictions ##
                      #########################
"""


# Load the GoEmotions classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
    truncation=True
)


def remove_thanking_phrases(text):
    """
    A helper function for classify emotion that removes all thanks (to avoid classifying it to gratitude)
    :param text: A string of the speech
    :return: cleaned version (removed thanks)
    """
    patterns = [
        r"\bthank(s| you| you all)?\b"
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.I)
    return text


def classify_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifying each speech with emotion.
    :param df: data frame with the column speech
    :return: a new data frame with a predicted emotion column
    """
    df = df.copy()

    # Define classification logic
    def get_top_emotion(text):
        # Converting text to a nlp object and filtering the tokens
        doc = nlp(remove_thanking_phrases(text).lower())

        tokens = [
            token.lemma_ for token in doc
            if token.is_alpha and token.pos_ in ALLOWED_POS
        ]

        # Classifying emotion or neutral if unsuccessful
        try:
            result = emotion_classifier(' '.join(tokens))
            return result[0][0]['label'] if result else "neutral"
        except Exception as e:
            print(f"Error processing text: {text[:30]}... -> {e}")
            return "error"

    # Apply classifier to each speech
    df['predicted_emotion'] = df['speech'].progress_apply(get_top_emotion)
    return df


"""
                      **********
                      ** Main **
                      **********
"""


if __name__ == "__main__":
    #
    # speeches_df = clean_presidential_speeches(r'Data\presidential_speeches.xlsx')
    # sentiment_results = speeches_df['speech'].apply(extract_sentiments)
    #
    # # Convert the series of dicts into a DataFrame with separate columns
    # sentiment_df = sentiment_results.apply(pd.Series)
    #
    # # Join the new sentiment columns to your original DataFrame
    # speeches_df = pd.concat([speeches_df, sentiment_df], axis=1)
    #
    # # Save the updated DataFrame
    # speeches_df.to_excel("speeches_with_sentiment.xlsx", index=False)
    #

    # speeches_df['topics'] = speeches_df['speech'].apply(classify_topic)
    # speeches_df.to_excel("speeches_with_topics_different_threshold.xlsx", index=False)

    # result_df = classify_emotion(speeches_df)
    # result_df.to_excel("speeches_with_emotions.xlsx", index=False)

    # Load all data
    speeches_with_sentiment_df = pd.read_excel(r'speeches_with_sentiment.xlsx')
    speeches_with_emotions_df = pd.read_excel(r'speeches_with_emotions_final.xlsx')
    speeches_with_topics_df = pd.read_excel(r'speeches_with_topics_different_threshold.xlsx')

    # Adjust sentiment labels if needed
    speeches_with_sentiment_df.loc[
        speeches_with_sentiment_df['positivity_score'].between(0.4, 0.6, inclusive='both'),
        'label'
    ] = 'NEUTRAL'

    # Merge all three DataFrames on the 'speech' column
    combined_df = speeches_with_emotions_df.merge(
        speeches_with_topics_df[['speech', 'topics']],
        on='speech',
        how='left'
    ).merge(
        speeches_with_sentiment_df[['speech', 'label', 'positivity_score']],
        on='speech',
        how='left'
    )

    # Save to Excel
    combined_df.to_excel("combined_predictions.xlsx", index=False)



