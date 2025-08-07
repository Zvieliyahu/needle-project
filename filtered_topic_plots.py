import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
import spacy
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

    # Use get with default 0 for missing columns
    positive = counts.get('positive', pd.Series(0, index=counts.index)).tolist()
    negative = counts.get('negative', pd.Series(0, index=counts.index)).tolist()
    neutral = counts.get('neutral', pd.Series(0, index=counts.index)).tolist()

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


def plot_party_speeches_by_decade(df: pd.DataFrame):
    # Ensure 'date' is datetime and create 'decade'
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['decade'] = (df['date'].dt.year // 10) * 10

    # Filter speeches about Black rights
    filtered_df = df[df[f'contains_{TOPIC}_keywords'] == True]

    # Group by Party and Decade
    counts = filtered_df.groupby(['decade', 'Party']).size().unstack(fill_value=0)

    # Prepare for plotting
    decades = counts.index.tolist()
    parties = counts.columns.tolist()

    x = range(len(decades))
    width = 0.35 if len(parties) == 2 else 0.25  # adjust width if more parties

    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors for common parties
    party_colors = {
        'Democratic': 'blue',
        'Republican': 'red'
    }

    for i, party in enumerate(parties):
        values = counts[party].tolist()
        bar_positions = [p + (i - len(parties) / 2) * width + width / 2 for p in x]
        ax.bar(bar_positions, values, width=width, label=party, color=party_colors.get(party, 'gray'))

    # Labels and ticks
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in decades], rotation=45)
    ax.set_ylabel('Number of Speeches')
    ax.set_title('Presidential Speeches About Black Rights by Party and Decade')
    ax.legend(title='Party')

    plt.tight_layout()
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


def plot_tfidf_word_cloud(df: pd.DataFrame, full_speeches_df: pd.DataFrame, text_column='speech'):
    # Combine speeches into documents
    target_docs = df[text_column].dropna().astype(str).tolist()
    background_docs = full_speeches_df[text_column].dropna().astype(str).tolist()

    # Prepare corpus: target + background
    corpus = target_docs + background_docs

    # Create labels to identify which docs belong to target
    n_target = len(target_docs)

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

    # Fit and transform corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Separate TF-IDF for target docs and background docs
    target_tfidf = tfidf_matrix[:n_target]
    background_tfidf = tfidf_matrix[n_target:]

    # Average TF-IDF scores for each term in target and background
    import numpy as np
    target_mean = np.asarray(target_tfidf.mean(axis=0)).ravel()
    background_mean = np.asarray(background_tfidf.mean(axis=0)).ravel()

    # Calculate "uniqueness" score: target_mean minus background_mean (or ratio)
    uniqueness = target_mean - background_mean  # could also do target_mean / (background_mean + 1e-9)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Create dictionary of words and uniqueness scores (only positive scores)
    unique_words = {word: score for word, score in zip(feature_names, uniqueness) if score > 0}

    # Generate word cloud weighted by uniqueness score
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=50
    ).generate_from_frequencies(unique_words)

    # Plot the word cloud
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('TF-IDF Weighted Word Cloud of Unique Words', fontsize=20)
    plt.show()
    plt.savefig("immigration/tf_idf_speeches")


PATH_IMMIGRATION = 'immigration/cutted_speeches_that_include_immigration_keywords.xlsx'
PATH_BLACK_RIGHTS = 'black rights/cutted_speeches_that_include_black_rights_keywords.xlsx'
PATH_WOMEN_RIGHTS = 'women rights/cutted_speeches_that_include_womens_rights_keywords.xlsx'
PATH_NATIVE_AMERICANS = 'native americans/cutted_speeches_that_include_native_americans_keywords.xlsx'
PATH_ORIGINAL_SPEECHES = 'Data/presidential_speeches.xlsx'
TOPIC = "native_americans"


def run_plots(path):
    plot_bar_by_century(pd.read_excel(path))
    plot_bar_by_party_and_positivity(pd.read_excel(path))
    plot_party_speeches_by_decade(pd.read_excel(path))
    plot_word_cloud(pd.read_excel(path))
    plot_tfidf_word_cloud(pd.read_excel(path), pd.read_excel(PATH_ORIGINAL_SPEECHES))


if __name__ == '__main__':
    run_plots(PATH_NATIVE_AMERICANS)
