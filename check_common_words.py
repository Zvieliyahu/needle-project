from cleanData import clean_presidential_speeches
import pandas as pd
from collections import Counter
import re
import nltk
from processData import topics

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
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_.lower() not in stop_words
    ]
    # Count word frequencies
    word_counts = Counter(tokens)

    # Get top `n` words
    top_words = word_counts.most_common(n)
    top_word_list = [word for word, count in top_words]

    matched_topics = {}
    for word, count in top_words:
        for topic, keywords in topics.items():
            if word in keywords:
                if topic in matched_topics.keys():
                    matched_topics[topic] += count
                else:
                    matched_topics[topic] = count


    # Format result: word1:count1, word2:count2, ...
    print(matched_topics)
    return ', '.join([f"{word}:{count}" for word, count in matched_topics.items()])


if __name__ == "__main__":
    speeches_df = clean_presidential_speeches(r'Data\presidential_speeches.xlsx')
    speeches_df.loc[:99, 'top_words'] = speeches_df.loc[:99, 'speech'].apply(get_top_words)
    speeches_df.to_excel("common_words3.xlsx", index=False)
