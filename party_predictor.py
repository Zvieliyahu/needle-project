from predictor_helper import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import streamlit as st
import numpy as np


def create_database_vectors(df: pd.DataFrame):
    """
    Function to create the database of vectors of party (that will be compared with)
    Uses K-Means algorithm to cluster the vectors of each party.
    :param df: Dataframe with the processed speeches.
    :return:
    """
    n_clusters = 30

    df_filtered = df[df['Party'].isin(['Democratic', 'Republican'])].copy()
    df_filtered = add_topic_columns(df_filtered)
    df_filtered = add_emotion_columns(df_filtered)
    df_filtered = add_label_columns(df_filtered)
    df_filtered = add_topic_sentiment_scores(df_filtered)
    df_filtered.to_excel('database_vectors.xlsx', index=False)
    df_filtered = pd.read_excel('database_vectors.xlsx')
    # Compute mean vectors of each party
    party_mean_vectors = df_filtered.groupby('Party')[FEATURE_COLUMNS].mean().reset_index()
    party_mean_vectors['Cluster'] = 'Mean'

    def compute_party_clusters(df_single_party, party_name):
        """
        Function to compute the clusters of a party.
        :param df_single_party: Dataframe with the processed speeches of a specific party.
        :param party_name: Name of the party.
        :return: List of clusters.
        """
        vectors = df_single_party[FEATURE_COLUMNS]
        if len(vectors) < n_clusters:
            raise ValueError(f"Not enough speeches for {party_name} to form {n_clusters} clusters.")

        kmeans = KMeans(n_clusters=n_clusters, random_state=128)
        kmeans.fit(vectors)

        cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURE_COLUMNS)
        cluster_centers['Party'] = party_name
        cluster_centers['Cluster'] = [f'Cluster_{i}' for i in range(n_clusters)]
        return cluster_centers

    df_dem = df_filtered[df_filtered['Party'] == 'Democratic']
    dem_clusters = compute_party_clusters(df_dem, 'Democratic')

    df_rep = df_filtered[df_filtered['Party'] == 'Republican']
    rep_clusters = compute_party_clusters(df_rep, 'Republican')

    # Combine everything (mean vectors + clusters)
    all_vectors = pd.concat([party_mean_vectors, dem_clusters, rep_clusters], ignore_index=True)
    all_vectors.to_csv("party_vectors.csv", index=False)


def predict_party(text: str):
    """
    Predicts the party of a speech.
    :param text: The text to predict - speech.
    :return: Party predicted of the speech, 5 most similar features to the most similar vector.
    """
    df_party = pd.read_csv('party_vectors.csv')

    # apply preprocessing on input
    df = pd.DataFrame({'speech': [text]})
    df = classify_emotion(df)
    df['topics'] = df['speech'].apply(classify_topic)
    df_sent = df['speech'].apply(assign_positivity_label)
    df_sent = df_sent.apply(pd.Series)
    df = pd.concat([df, df_sent], axis=1)
    df = add_topic_columns(df)
    df = add_emotion_columns(df)
    df = add_label_columns(df)
    df = add_topic_sentiment_scores(df)
    input_vector = df[FEATURE_COLUMNS]

    csv_vectors = df_party[FEATURE_COLUMNS].values
    vector = np.array(input_vector).reshape(1, -1)

    # Compute cosine similarities and find the most similar vector
    similarities = cosine_similarity(vector, csv_vectors)[0]
    most_similar_index = np.argmax(similarities)
    most_similar_row = df_party.iloc[most_similar_index]
    matched_vector = most_similar_row[FEATURE_COLUMNS].values.astype(float)
    input_values = input_vector.values.flatten().astype(float)
    differences = abs(input_values - matched_vector)
    # Get 5 most similar features to the most similar vector
    top_features_idx = differences.argsort()[:5]
    top_features = [FEATURE_COLUMNS[i] for i in top_features_idx]
    return most_similar_row["Party"], top_features


def misclassification_loss(df):
    """
    Calculates the misclassification loss between the party column in the input dataframe
    and the predicted_party column in the input dataframe.
    :param df: Dataframe containing party column and predicted_party column
    :return:
    """
    # Ensure columns exist
    if 'Party' not in df.columns or 'predicted_party' not in df.columns:
        raise ValueError("DataFrame must contain 'Party' and 'predicted_party' columns")

    # Count mismatches
    df_valid = df[df['predicted_party'].notna() & (df['predicted_party'].astype(str).str.strip() != '')]
    incorrect = (df_valid['Party'] != df_valid['predicted_party']).sum()
    total = len(df_valid)
    print('Incorrect predictions: ', incorrect)
    print('Incorrect predictions \\ total (percentage): ', incorrect / total)


if __name__ == '__main__':
    """
    Streamlit application of Party Prediction
    """
    st.title("Political Party Predictor")
    st.write("Enter a political speech or text below to predict whether it aligns more with the"
             " **Democratic** or **Republican** party.")

    text_input = st.text_area("Input speech text here:", height=200)

    if st.button("Predict"):
        if text_input.strip():
            with st.spinner("Analyzing..."):
                party, top_features = predict_party(text_input)
            st.success(f"Predicted Party: **{party}**")
            st.markdown("Top 5 Most Similar Features:")
            for feature in top_features:
                st.write(f"• {feature}")
        else:
            st.warning("Please enter some text to analyze.")
