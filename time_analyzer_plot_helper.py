import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.corpus import stopwords
import spacy
from process_helper import UNRELATED_TOPIC_WORDS
import re
import numpy as np
from PIL import Image

# GLOBALS #
TOPIC_PLOT_CAPTION = ("This plot shows the number of speeches of each topic by decade.\n"
                      "Each topic was being processed from the main data set and was categorized using termed rules.")

# Load stop words (English)
stop_words = set(stopwords.words('english')) | set(UNRELATED_TOPIC_WORDS)

# Load spaCy English model once
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
for word in UNRELATED_TOPIC_WORDS:
    nlp.vocab[word].is_stop = True


def plot_tfidf_word_cloud(df: pd.DataFrame,
                          full_speeches_df: pd.DataFrame,
                          text_column='speech',
                          date_column='date',
                          topic='',
                          start_year=None,
                          end_year=None,
                          output_path=None,):
    """
    Creating a word cloud for a data frame using TF-IDF.
    :param topic:
    :param df:
    :param full_speeches_df:
    :param text_column:
    :param date_column:
    :param start_year:
    :param end_year:
    :param output_path:
    :return:
    """

    def remove_punctuation(text):
        return re.sub(r'[^\w\s]', '', text)

    # Convert date columns
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    full_speeches_df[date_column] = pd.to_datetime(full_speeches_df[date_column], errors='coerce')

    # Filter target df
    if start_year is not None:
        df = df[df[date_column].dt.year >= start_year]
    if end_year is not None:
        df = df[df[date_column].dt.year <= end_year]

    # Prepare texts
    target_docs = df[text_column].dropna().astype(str).apply(remove_punctuation).tolist()
    background_raw = full_speeches_df[text_column].dropna().astype(str).apply(remove_punctuation).tolist()

    # Combine corpora
    corpus = target_docs + background_raw
    n_target = len(target_docs)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Split matrices
    target_tfidf = tfidf_matrix[:n_target]
    background_tfidf = tfidf_matrix[n_target:]

    # Mean TF-IDF
    target_mean = np.asarray(target_tfidf.mean(axis=0)).ravel()
    background_mean = np.asarray(background_tfidf.mean(axis=0)).ravel()

    # Compute uniqueness
    uniqueness = target_mean - background_mean
    feature_names = vectorizer.get_feature_names_out()
    unique_words = {word: score for word, score in zip(feature_names, uniqueness) if score > 0}

    # Word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=50
    ).generate_from_frequencies(unique_words)

    # Plot
    plt.figure(figsize=(15, 7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    title = f'{topic} Word Cloud ({start_year}–{end_year})' if start_year or end_year else 'TF-IDF Weighted Word Cloud'
    plt.title(title, fontsize=20)
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()


def add_annotations(ax, pivot):
    """
    Add hardcoded annotations to the speeches per decade by topic plot.

    Parameters:
        ax (matplotlib.axes.Axes): The axes to annotate.
        pivot (DataFrame): Pivoted data (decade x topic).
    """
    if "Black Rights" in pivot.columns and 1960 in pivot.index:
        ax.annotate(
            "Civil Rights Movement",
            xy=(1960, pivot.loc[1960, "Black Rights"]),
            xytext=(1950, pivot.loc[1960, "Black Rights"]+5),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=9, ha="center"
        )
    if "Black Rights" in pivot.columns and 1860 in pivot.index:
        ax.annotate(
            "Abolition of Slavery",
            xy=(1860, pivot.loc[1860, "Black Rights"]),
            xytext=(1850, pivot.loc[1860, "Black Rights"]+3),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=9, ha="center"
        )
    if "Immigration" in pivot.columns and 1880 in pivot.index:
        ax.annotate(
            "Burlingame Treaty\n&\nChinese Exclusion Act",
            xy=(1880, pivot.loc[1880, "Immigration"]),
            xytext=(1880, pivot.loc[1880, "Immigration"]+3),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=9, ha="center"
        )
    if "Immigration" in pivot.columns and 2010 in pivot.index:
        ax.annotate(
            "Contemporary\nImmigration Debate",
            xy=(2010, pivot.loc[2010, "Immigration"]),
            xytext=(1990, pivot.loc[2010, "Immigration"]-3),
            arrowprops=dict(facecolor='black', arrowstyle="->"),
            fontsize=9, ha="center"
        )


def plot_speeches_per_decade_by_topic(topic_dfs: dict):
    """
    Plots a line graph showing the number of speeches on each topic per decade.

    Parameters:
        topic_dfs (dict): Dictionary where keys are topic names (str)
                          and values are DataFrames with a 'date' column.
    """
    all_counts = []

    for topic, df in topic_dfs.items():
        # Ensure 'date' column is datetime
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])

        # Create 'decade' column (e.g., 1860, 1930)
        df['decade'] = (df['date'].dt.year // 10) * 10

        # Count speeches per decade for this topic
        decade_counts = df.groupby('decade').size().reset_index(name='count')
        decade_counts['topic'] = topic
        all_counts.append(decade_counts)

    # Combine all topic counts into one DataFrame
    combined = pd.concat(all_counts)
    pivot = combined.pivot(index='decade', columns='topic', values='count').fillna(0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in pivot.columns:
        ax.plot(pivot.index, pivot[column], label=column)

    fig.suptitle("Number of Speeches per Topic by Decade", fontsize=14, y=0.98)
    ax.text(
        0.01, 1.02,
        TOPIC_PLOT_CAPTION,
        transform=ax.transAxes,
        fontsize=10, color="gray", ha="left", va="bottom"
    )

    ax.set_xlabel('Decade')
    ax.set_ylabel('Number of Speeches')
    ax.legend(title='Topic')
    ax.grid(True)

    add_annotations(ax, pivot)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig("topics_of_presidential_speeches_by_decade.png", bbox_inches="tight")


def combine_images_side_by_side(paths, output_path=None, subtitle=""):
    """
    Combine multiple images side by side.

    :param subtitle: subtitle text to add below the title
    :param paths: list of file paths to images
    :param output_path: optional path to save the combined image
    :return: combined PIL.Image object
    """
    if not paths:
        raise ValueError("No image paths provided.")

    # Open all images
    images = [Image.open(p) for p in paths]

    # Match all heights to the height of the first image
    base_height = images[0].height
    resized_images = []
    for img in images:
        if img.height != base_height:
            new_width = int(img.width * (base_height / img.height))
            img = img.resize((new_width, base_height))
        resized_images.append(img)

    # Calculate total width and max height
    total_width = sum(img.width for img in resized_images)
    max_height = max(img.height for img in resized_images)

    # Create new blank image
    new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    # Paste images next to each other
    x_offset = 0
    for img in resized_images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width

    # Adding subtitle
    if subtitle:
        new_img.suptitle()

    # Save or show
    if output_path:
        new_img.save(output_path)
    else:
        new_img.show()

    return new_img

