import wordcloud
import matplotlib.pyplot as plt
import pandas as pd
from process_helper import *
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

# Define your custom list
ANOTHER_UNRELATED_TOPIC_WORDS = ['without', 'part', 'said', 'years', 'year', 'first', 'two', 'cut',
                                 'weve', 'upon', 'one', 'now', 'may', 'day', 'made', 'every', 'thats',
                                 'americans', 'going']

# Define your own irrelevant topic words
non_topic_words = ['government', 'people', 'nation', 'country', 'america', 'american']

# Mark them as stop words in spaCy
for word in non_topic_words:
    lex = nlp.vocab[word]
    lex.is_stop = True

# Combine all stopwords
combined_stopwords = set(wordcloud.STOPWORDS) | set(UNRELATED_TOPIC_WORDS) | \
                     set(ANOTHER_UNRELATED_TOPIC_WORDS) | set(non_topic_words) | \
                     set(nlp.Defaults.stop_words)
def create_word_cloud(df : pd.DataFrame):
    years = [(1789, 1899), (1900, 1949), (1950, 1979), (1980, 2021)]
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df['year'] = df['date'].dt.year
    for start, end in years:
        for party in ['Democratic', 'Republican']:
            # Filter data for the current year range and party
            filtered_df = df[(start <= df["year"]) & (df["year"] <= end) & (df["Party"] == party)]

            # Concatenate all speeches into a single string
            text = " ".join(speech for speech in filtered_df['speech'])

            # Generate the word cloud
            wc = wordcloud.WordCloud(width=800, height=400, background_color='white', stopwords=combined_stopwords).generate(
                text)

            # Plot the word cloud
            plt.figure(figsize=(24, 12))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(f'{party} Speeches Word Cloud {start}-{end}', fontsize=30, fontweight='bold')

            # Save the image
            plt.savefig(f"wordcloud_{party}_{start}-{end}.png")
            plt.clf()  # Clear the plot for the next word cloud

create_word_cloud(pd.read_excel("combined_predictions.xlsx"))