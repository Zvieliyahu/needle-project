# import pandas as pd
# import re
# from predictor_helper import *
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
# import matplotlib.pyplot as plt
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#
#
#
#
#
#
# # Step 1: Group speeches per president
# president_docs = df.groupby("President")["speech"].apply(lambda texts: " ".join(texts)).reset_index()
#
# # Step 2: Custom stopwords (expanded)
# custom_stopwords = {
#     "state", "states", "united", "united states", "u.s.", "us", "usa",
#     "government", "administration", "congress", "nation", "national", "country", "federal",
#     "thank", "thanks", "thank you", "today", "everyone", "ladies", "gentlemen",
#     "america", "american", "americans", "we", "our", "ours", "my", "me", "i", "us",
#     "know", "thats", "going", "didnt", "youre", "said", "dont", "years", "trump", "tens", "running"
#     "president", "people", "leader", "leadership", "public", "world", "every", "must", "theyre", "weve", "im", "new"
# }
#
# all_stopwords = list(ENGLISH_STOP_WORDS.union(custom_stopwords).union(UNRELATED_TOPIC_WORDS))
#
# # Step 3: TF-IDF Vectorizer
# vectorizer = TfidfVectorizer(
#     lowercase=True,
#     stop_words=all_stopwords,
#     max_features=5000,
#     ngram_range=(1, 2)
# )
#
# # Step 4: Vectorization
# tfidf_matrix = vectorizer.fit_transform(president_docs["speech"])
# feature_names = vectorizer.get_feature_names_out()
#
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(),
#                         index=president_docs["President"],
#                         columns=feature_names)
#
# # Step 5: Show top terms per president
# percentile_cutoff = 90  # You can tune this
#
# for president, row in tfidf_df.iterrows():
#     threshold = row.quantile(percentile_cutoff / 100.0)
#     strong_terms = row[row > threshold].sort_values(ascending=False)
#
#     if not strong_terms.empty:
#         print(f"\n🔹 Strong terms for {president} (above {percentile_cutoff}th percentile = {threshold:.4f}):")
#         print(strong_terms)
#     else:
#         print(f"\n🔹 No strong terms for {president} above the {percentile_cutoff}th percentile")
#
# df = pd.read_excel("combined_predictions.xlsx")
# # pd.set_option('display.max_columns', None)
# # print(data.head(10))
# df['date'] = pd.to_datetime(df['date'])
#
# # Extract decade from the year
# df['decade'] = (df['date'].dt.year // 10) * 10
# df['decade'] = df['decade'].astype(str) + 's'
#
# total_per_decade = df.groupby('decade').size().to_dict()
#
# # Count speeches per label per decade
# label_counts = df.groupby(['decade', 'label']).size().reset_index(name='count')
#
# # Unique decades and labels
# decades = sorted(label_counts['decade'].unique())
# labels = sorted(label_counts['label'].unique())  # consistent order
#
# # Build a nested dictionary: {decade: {label: percentage}}
# data_percent = {decade: {label: 0 for label in labels} for decade in decades}
# for _, row in label_counts.iterrows():
#     decade = row['decade']
#     label = row['label']
#     count = row['count']
#     total = total_per_decade[decade]
#     data_percent[decade][label] = count / total * 100  # Convert to percentage
#
# # Bar width and x locations
# bar_width = 0.2
# x = range(len(decades))
#
# # Plotting
# plt.figure(figsize=(10, 6))
#
# for i, label in enumerate(labels):
#     percentages = [data_percent[decade][label] for decade in decades]
#     offset = [xi + i * bar_width for xi in x]
#     plt.bar(offset, percentages, width=bar_width, label=label)
#
# # X-axis ticks centered
# middle_positions = [xi + (len(labels)-1)*bar_width/2 for xi in x]
# plt.xticks(middle_positions, decades)
#
# # Labels and title
# plt.title('Percentage of Predicted Labels by Decade')
# plt.xlabel('Decade')
# plt.ylabel('Percentage of Speeches')
# plt.legend(title='Label')
# plt.tight_layout()
# plt.savefig("Lable over time.png")
# data = pd.read_excel("Data\presidential_speeches.xlsx")
# # pd.set_option('display.max_columns', None)
# # print(data.head())
# df = pd.DataFrame(data)
#
# # ---- Step 2: Preprocessing ----
# # Convert year to decade
# df['date'] = pd.to_datetime(df['date'])
#
# # Step 2: Extract year and decade
# df['year'] = df['date'].dt.year
# df['decade'] = (df['year'] // 10) * 10
# df = df[df['Party'].isin(['Democratic', 'Republican'])]
#
# # Normalize text to lowercase
# df['speech'] = df['speech'].str.lower()
#
# # ---- Step 3: Count 'we' and 'i' ----
# df['we_count'] = df['speech'].str.count(r'\bwe\b')
# df['i_count'] = df['speech'].str.count(r'\bi\b')
#
# # ---- Step 4: Group by party and decade ----
# grouped = df.groupby(['Party', 'decade'])[['we_count', 'i_count']].sum().reset_index()
#
# # ---- Step 5: Visualization ----
#
# # Set up the figure and subplots
# fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
#
# # Plot for 'we'
# for party in grouped['Party'].unique():
#     party_data = grouped[grouped['Party'] == party]
#     axs[0].plot(party_data['decade'], party_data['we_count'], marker='o', label=party)
#
# axs[0].set_title("Use of 'we' Over Time by Party")
# axs[0].set_xlabel("Decade")
# axs[0].set_ylabel("Count of 'we'")
# axs[0].legend()
# axs[0].grid(True)
#
# # Plot for 'i'
# for party in grouped['Party'].unique():
#     party_data = grouped[grouped['Party'] == party]
#     axs[1].plot(party_data['decade'], party_data['i_count'], marker='o', label=party)
#
# axs[1].set_title("Use of 'I' Over Time by Party")
# axs[1].set_xlabel("Decade")
# axs[1].set_ylabel("Count of 'I'")
# axs[1].legend()
# axs[1].grid(True)
#
# # Show the plots
# plt.tight_layout()
# plt.savefig("WeAndI.png")
# pd.set_option('display.max_columns', None)
# print(df.head(10))
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from process_helper import *
from nltk.corpus import stopwords
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({
    "im", "doesnt", "dont", "ive", "youre", "youve", "weve", "theyre", "isnt",
    "arent", "wont", "wouldnt", "lets", "thats", "cant", "wasnt", "hasnt",
    "hadnt", "havent", "isnt", "aint", "shouldnt", "couldnt", "wheres", "whats", "whos", "bin",
    "iraq", "qaeda", "assad", "laden", "osama", "andwhereas", "within", "aforesaid", "americaa", "whereof", "persons",
    "said", "proclamationwhereas"
}).union(UNRELATED_TOPIC_WORDS)
def create_word_cloud_obama_economy():
    df = pd.read_excel('emotion_and_positivity_predictions.xlsx')
    # Ensure stopwords are available

    # Add custom words to exclude (contractions, filler)

    # Load your dataset
    # Replace this with your actual CSV or DataFrame load step
    # Example:
    # df = pd.read_csv("your_dataset.csv")

    # Filter to only include Barack Obama's speeches
    obama_df = df[df['President'] == 'Barack Obama']

    # Filter those with "Economy" in the topics
    obama_economy = obama_df[obama_df['topics'].str.contains("Economy", case=False, na=False)]

    # Other speeches (not Obama or not Economy)
    other_speeches = df[~df.index.isin(obama_economy.index)]

    # Preprocess function to clean text
    def preprocess(text):
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())  # Remove punctuation and lowercase
        tokens = [word for word in text.split() if word not in custom_stopwords and len(word) > 2]
        return " ".join(tokens)

    # Apply preprocessing
    obama_texts = obama_economy['speech'].dropna().apply(preprocess)
    other_texts = other_speeches['speech'].dropna().apply(preprocess)

    # Combine into a corpus for TF-IDF
    corpus = list(obama_texts) + list(other_texts)
    labels = ['obama'] * len(obama_texts) + ['other'] * len(other_texts)

    # Perform TF-IDF
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()

    # Get average TF-IDF per word in Obama vs Other
    import numpy as np
    obama_mask = np.array(labels) == 'obama'
    other_mask = ~obama_mask

    # Average TF-IDF values for each group
    obama_tfidf_mean = X[obama_mask].mean(axis=0).A1
    other_tfidf_mean = X[other_mask].mean(axis=0).A1

    # Calculate difference
    diff = obama_tfidf_mean - other_tfidf_mean

    # Create a dictionary of top N unique words in Obama's Economy speeches
    top_n = 100
    top_indices = diff.argsort()[-top_n:]
    word_scores = {features[i]: diff[i] for i in top_indices if diff[i] > 0}

    # Generate Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=30).generate_from_frequencies(
        word_scores)

    # Display
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Top Unique Words in Obama's Economy Speeches")
    plt.savefig("obama_economy_wordcloud.png", format='png')


def create_word_cloud_angry_years():

    df = pd.read_excel('emotion_and_positivity_predictions.xlsx')

    # Assuming you have the DataFrame `df` with the datadf = pd.read_excel('emotion_and_positivity_predictions.xlsx')
    #
    # Step 1: Filter angry speeches between 1840-1910
    angry_speeches = df[(df['predicted_emotion'] == 'anger') & (df['from'] >= 1840) & (df['from'] <= 1910)]

    # Step 2: Filter out non-angry speeches (the rest of the speeches)
    other_speeches = df[~df.index.isin(angry_speeches.index)]

    # Step 3: Preprocess function to clean the text
    def preprocess(text):
        # Remove non-alphabetical characters and convert to lowercase
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        # Remove stopwords and words with length <= 2
        tokens = [word for word in text.split() if word not in custom_stopwords and len(word) > 2]
        return " ".join(tokens)

    # Apply preprocessing to the speeches
    angry_texts = angry_speeches['speech'].dropna().apply(preprocess)
    other_texts = other_speeches['speech'].dropna().apply(preprocess)

    # Step 4: Combine both angry and other speeches for TF-IDF comparison
    corpus = list(angry_texts) + list(other_texts)
    labels = ['anger'] * len(angry_texts) + ['other'] * len(other_texts)

    # Step 5: Perform TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(corpus)
    features = vectorizer.get_feature_names_out()

    # Step 6: Get average TF-IDF per word for angry vs other
    angry_mask = np.array(labels) == 'anger'
    other_mask = ~angry_mask

    # Average TF-IDF values for each group
    angry_tfidf_mean = X[angry_mask].mean(axis=0).A1
    other_tfidf_mean = X[other_mask].mean(axis=0).A1

    # Step 7: Calculate the difference in TF-IDF scores between angry speeches and other speeches
    diff = angry_tfidf_mean - other_tfidf_mean

    # Step 8: Get top N unique words in angry speeches (those with higher TF-IDF)
    top_n = 100  # You can change this number based on how many words you want
    top_indices = diff.argsort()[-top_n:]  # Get the indices of the top N words

    # Create a dictionary with words and their TF-IDF difference values
    word_scores = {features[i]: diff[i] for i in top_indices if diff[i] > 0}

    # Step 9: Generate Word Cloud for the unique words in angry speeches
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=30).generate_from_frequencies(word_scores)

    # Step 10: Display the word cloud
    plt.figure(figsize=(15, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Top Unique Words in Angry Speeches (1840-1910)")
    plt.savefig("angry_speeches_wordcloud.png", format='png')


create_word_cloud_angry_years()