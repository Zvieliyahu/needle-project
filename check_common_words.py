from cleanData import clean_presidential_speeches
import pandas as pd
from collections import Counter
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy

# Load stop words (English)
stop_words = set(stopwords.words('english'))

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")


def get_top_words(text, n=15):
    """
    Counting the most common lemmatized words without stopwords using spaCy.
    :param text: Raw text input
    :param n: Number of top words to return
    :return: Formatted string of top words and their counts
    """
    # Process the text with spaCy
    doc = nlp(text.lower())

    # Filter: remove stopwords, punctuation, numbers, and non-alphabetic tokens
    tokens = [token.lemma_ for token in doc
              if token.is_alpha and not token.is_stop]

    # Count word frequencies
    word_counts = Counter(tokens)

    # Get top `n` words
    top_words = word_counts.most_common(n)

    # Format result: word1:count1, word2:count2, ...
    return ', '.join([f"{word}:{count}" for word, count in top_words])


if __name__ == "__main__":
    speeches_df = clean_presidential_speeches(r'Data\presidential_speeches.xlsx')
    speeches_df['top_words'] = speeches_df['speech'].apply(get_top_words)
    speeches_df.to_excel("common_words2.xlsx", index=False)

