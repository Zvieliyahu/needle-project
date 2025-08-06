import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import spacy
from collections import Counter
from process_helper import UNRELATED_TOPIC_WORDS, ALLOWED_POS


def plot_bar_by_century(df: pd.DataFrame):
    # 1. Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # 2. Create a 2-decade bin column (e.g., 1860s–1879s, 1880s–1899s, etc.)
    df['decade_bin'] = (df['date'].dt.year // 100) * 100

    # 3. Group by decade bin and sentiment label, then count speeches
    sentiment_counts = df.groupby(['decade_bin', 'label']).size().unstack(fill_value=0)

    # Ensure all three sentiment types exist in columns
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0

    # 4. Sort by decade
    sentiment_counts = sentiment_counts.sort_index()

    # 5. Plot
    colors = {'positive': 'blue', 'neutral': 'black', 'negative': 'red'}
    sentiment_counts[['positive', 'neutral', 'negative']].plot(
        kind='bar',
        color=[colors['positive'], colors['neutral'], colors['negative']],
        figsize=(12, 6)
    )

    plt.title("Sentiment Distribution of Speeches by 2-Decade Periods")
    plt.xlabel("Decade Period")
    plt.ylabel("Number of Speeches")
    plt.legend(title="Sentiment")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_bar_by_party_and_positivity(df: pd.DataFrame):
    counts = df.groupby(['Party', 'label']).size().unstack(fill_value=0)
    labels = counts.index.tolist()
    positive = counts['positive'].tolist()
    negative = counts['negative'].tolist()
    neutral = counts['neutral'].tolist()

    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar([p - width for p in x], positive, width=width, color='blue', label='Positive')
    ax.bar(x, negative, width=width, color='red', label='Negative')
    ax.bar([p + width for p in x], neutral, width=width, color='black', label='Neutral')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of Speeches')
    ax.set_title('Sentiment Counts by Party')
    ax.legend()

    plt.show()


# Load stop words (English)
stop_words = set(stopwords.words('english')) | set(UNRELATED_TOPIC_WORDS)

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
for word in UNRELATED_TOPIC_WORDS:
    nlp.vocab[word].is_stop = True


def clean_text(text):
    doc = nlp(text.lower())

    # Filter: remove stopwords, punctuation, numbers, and non-alphabetic tokens
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_.lower() not in stop_words
    ]

    return ' '.join(tokens)


def plot_word_cloud(df: pd.DataFrame):
    all_speeches = clean_text(" ".join(df['speech'].dropna().astype(str).tolist()))

    # Create the word cloud object
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=200
    ).generate(all_speeches)

    # Plot the word cloud
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of All Speeches Combined', fontsize=20)
    plt.show()


if __name__ == '__main__':
    plot_bar_by_century(pd.read_excel('immigration/cutted_speeches_that_include_immigration_keywords.xlsx'))
    plot_bar_by_party_and_positivity(pd.read_excel('immigration/cutted_speeches_that_include_immigration_keywords.xlsx'))
    plot_word_cloud(pd.read_excel('immigration/cutted_speeches_that_include_immigration_keywords.xlsx'))
