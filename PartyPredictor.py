import pandas as pd
import numpy as np
from process_helper import *
from check_common_words import *
from process_data import *
from predictor_helper import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import streamlit as st
count = 0
df_party = pd.read_csv('party_vectors.csv')


# Get clean data output vectors that represents each party
def create_database_vectors(df : pd.DataFrame):
    N_CLUSTERS = 70

    df_filtered = df[df['Party'].isin(['Democratic', 'Republican'])].copy()

    df_filtered = add_topic_columns(df_filtered)
    df_filtered = add_emotion_columns(df_filtered)
    df_filtered = add_label_columns(df_filtered)

    df_filtered.to_excel('database_vectors.xlsx', index=False)
    # Compute mean vectors by party
    party_mean_vectors = df_filtered.groupby('Party')[FEATURE_COLUMNS].mean().reset_index()
    party_mean_vectors['Cluster'] = 'Mean'  # Mark these rows as mean vectors

    def compute_party_clusters(df_party, party_name):
        vectors = df_party[FEATURE_COLUMNS]
        if len(vectors) < N_CLUSTERS:
            raise ValueError(f"Not enough speeches for {party_name} to form {N_CLUSTERS} clusters.")

        kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
        kmeans.fit(vectors)

        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURE_COLUMNS)
        cluster_centers['Party'] = party_name
        cluster_centers['Cluster'] = [f'Cluster_{i}' for i in range(N_CLUSTERS)]
        return cluster_centers

    df_dem = df_filtered[df_filtered['Party'] == 'Democratic']
    dem_clusters = compute_party_clusters(df_dem, 'Democratic')

    df_rep = df_filtered[df_filtered['Party'] == 'Republican']
    rep_clusters = compute_party_clusters(df_rep, 'Republican')

    # Combine everything (mean vectors + clusters)
    all_vectors = pd.concat([party_mean_vectors, dem_clusters, rep_clusters], ignore_index=True)

    # Save to single CSV file
    all_vectors.to_csv("party_vectors.csv", index=False)


# Predicts a party for a string
def predict_party(text : str):
    # Maybe Preprocess
    global count, df_party
    # add the topics of the given text
    df = pd.DataFrame({'speech': [text]})
    df = classify_emotion(df)
    df['topics'] = df['speech'].apply(get_top_words)
    df_sent = df['speech'].apply(extract_sentiments)
    df_sent = df_sent.apply(pd.Series)
    df = pd.concat([df, df_sent], axis=1)
    df = add_topic_columns(df)
    # calculate the emotion
    df = add_emotion_columns(df)
    # label & positivity
    df = add_label_columns(df)
    input_vector = df[FEATURE_COLUMNS]
    #print(input_vector.head())
    # Calc the similarty from the csv

    csv_vectors = df_party[FEATURE_COLUMNS].values

    # Step 4: Convert your input vector to correct shape
    vector = np.array(input_vector).reshape(1, -1)

    # Step 5: Compute cosine similarities
    similarities = cosine_similarity(vector, csv_vectors)[0]

    # Step 6: Find the index of the most similar vector
    most_similar_index = np.argmax(similarities)

    # Step 7: Get the most similar row
    most_similar_row = df_party.iloc[most_similar_index]

    # Step 8: Print or use the result
    # print("Most similar row:")
    # print(most_similar_row["Party"])
    # print(f"Cosine similarity: {similarities[most_similar_index]:.4f}")
    count += 1
    if(count % 40 == 0):
        print(count)
    return most_similar_row["Party"]


def test_loss(input_vector):
    input_vector = input_vector[FEATURE_COLUMNS]
    csv_vectors = df_party[FEATURE_COLUMNS].values

    # Step 4: Convert your input vector to correct shape
    vector = np.array(input_vector).reshape(1, -1)

    # Step 5: Compute cosine similarities
    similarities = cosine_similarity(vector, csv_vectors)[0]

    # Step 6: Find the index of the most similar vector
    most_similar_index = np.argmax(similarities)

    # Step 7: Get the most similar row
    most_similar_row = df_party.iloc[most_similar_index]

    # Step 8: Print or use the result
    # print("Most similar row:")
    # print(most_similar_row["Party"])
    # print(f"Cosine similarity: {similarities[most_similar_index]:.4f}")
    return most_similar_row["Party"]
# loss for every wrong guess on the database it trained on
def misclassification_loss(df):
    # Ensure columns exist
    if 'Party' not in df.columns or 'predicted_party' not in df.columns:
        raise ValueError("DataFrame must contain 'Party' and 'predicted_party' columns")

    # Count mismatches
    df_valid = df[df['predicted_party'].notna() & (df['predicted_party'].astype(str).str.strip() != '')]

    incorrect = (df_valid['Party'] != df_valid['predicted_party']).sum()
    total = len(df_valid)
    print('Incorrect predictions: ', incorrect)
    print('Incorrect predictions \ total: ', incorrect / total)

# create_database_vectors(pd.read_excel("combined_predictions.xlsx"))
# # df = clean_presidential_speeches('Data\presidential_speeches.xlsx')
# # df = df[df['Party'].isin(['Democratic', 'Republican'])]
# # df['speech'] = df['speech'].apply(remove_thanking_phrases)
# df = pd.read_excel("database_vectors.xlsx")
# # df['predicted_party'] = df['speech'].head(50).apply(predict_party)
# df['predicted_party'] = df.apply(test_loss, axis=1)
#
# misclassification_loss(df)
# df.to_excel("prediction_result.xlsx", index=False)
st.title("Political Party Predictor")

st.write("Enter a political speech or text below to predict whether it aligns more with the **Democratic** or **Republican** party.")

text_input = st.text_area("Input speech text here:", height=200)

if st.button("Predict"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            party = predict_party(text_input)
        st.success(f"Predicted Party: **{party}**")
    else:
        st.warning("Please enter some text to analyze.")