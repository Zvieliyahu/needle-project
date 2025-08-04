from cleanData import clean_presidential_speeches
import pandas as pd
from collections import Counter
import re
import nltk
from processData import topics
from typing import Dict, List

nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy

non_topic_words = [
    # Your previous words:
    "go", "think", "want", "know", "say", "good", "lot", "get", "like",
    "thank", "time", "people", "work", "today", "thing", "great", "come",
    "look", "way", "right", "help", "need", "make", "well", "let", "tell",
    "see", "take", "try", "keep", "talk", "ask", "use", "give", "put",
    "feel", "seem", "leave", "mean", "start", "call", "show", "really",
    "big", "year", "new", "last", "many", "still", "find", "even",

    # Additional filler/generic/common words often not topic-specific:
    "also", "much", "much", "very", "always", "never", "quite", "just",
    "lot", "some", "most", "more", "most", "better", "better", "much",
    "well", "today", "tonight", "okay", "ok", "yes", "no", "let's", "us",
    "got", "getting", "did", "didn't", "does", "doesn't", "do", "don't",
    "will", "would", "could", "should", "might", "must", "can", "shall",
    "probably", "maybe", "actually", "basically", "seriously", "honestly",
    "definitely", "literally", "sure", "right", "okay", "alright", "hey",
    "hello", "hi", "well", "oh", "um", "uh", "yeah", "okay", "anyway",
    "anyways", "however", "therefore", "thus", "meanwhile", "although",
    "though", "yet", "still", "actually", "simply", "quite", "pretty",
    "rather", "soon", "often", "always", "never", "sometimes", "usually",
    "again", "already", "soon", "later", "ago",

    # Common pronouns (sometimes too generic for topic detection):
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",

    # Common conjunctions and prepositions that spaCy usually removes but can add:
    "and", "or", "but", "if", "because", "while", "as", "until", "when",
    "where", "after", "before", "since", "though", "although", "nor",
    "so", "than", "whether",
    
    # Other words we've missed
    "president", "go", "united", "states", "america", "country", "say", "nation", "world", "american"
]

# Load stop words (English)
stop_words = set(stopwords.words('english')) | set(non_topic_words)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
for word in non_topic_words:
    nlp.vocab[word].is_stop = True

# GLOBALS:
FIRST_THRESHOLD = 0.6
SECOND_THRESHOLD = 0.8


def predict_topics(words: Dict) -> str:
    if not words:
        return "None"

    total_count = 0
    for c in words.values():
        total_count += c

    normalized_count = {}
    most_common_topic = None
    for topic, count in words.items():
        normalized_count[topic] = count / total_count
        # Getting most common topic for prediction
        if most_common_topic is None:
            most_common_topic = topic, normalized_count[topic]
        elif most_common_topic[1] < normalized_count[topic]:
            most_common_topic = topic, normalized_count[topic]

    # If one dominant topic:
    if most_common_topic[1] > FIRST_THRESHOLD:
        return most_common_topic[0]

    # If no dominant topic returning most dominant topics:
    topic_coverage = 0
    topics = []
    while topic_coverage < SECOND_THRESHOLD:
        max_topic = max(normalized_count, key=normalized_count.get)
        topics.append(max_topic)
        topic_coverage += normalized_count[max_topic]
        del normalized_count[max_topic]
    return ', '.join([f"{word}" for word in topics])


def get_top_words(text, n=30):
    """
    Counting the most common lemmatized words without stopwords using spaCy.
    :param text: Raw text input
    :param n: Number of top words to return
    :return: Formatted string of top words and their counts
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
    # top_word_list = [word for word, count in top_words]
    # print(top_word_list)
    matched_topics = {}
    for word, count in top_words:
        for topic, keywords in topics.items():
            if word in keywords:
                if topic in matched_topics.keys():
                    matched_topics[topic] += count
                else:
                    matched_topics[topic] = count


    # Format result: word1:count1, word2:count2, ...
    # return ', '.join([f"{word}:{count}" for word, count in matched_topics.items()])

    # Predicting the subject:
    return predict_topics(matched_topics)


if __name__ == "__main__":
    speeches_df = clean_presidential_speeches(r'Data\presidential_speeches.xlsx')
    speeches_df['topics'] = speeches_df['speech'].apply(get_top_words)
    speeches_df.to_excel("speeches_with_topics.xlsx", index=False)
