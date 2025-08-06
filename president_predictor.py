import pandas as pd
from predictor_helper import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
df_president = pd.read_csv('presidents_vectors.csv')


def create_presidents_vectors(df):
    # df_filtered = df.copy()
    # # Create vectors for each row - same as party
    # df_filtered = add_topic_columns(df_filtered)
    # df_filtered = add_emotion_columns(df_filtered)
    # df_filtered = add_label_columns(df_filtered)
    # df_filtered.to_excel('database_vectors_president.xlsx', index=False)
    # # Maybe should be mean in other way
    # president_mean_vectors = df_filtered.groupby('President')[FEATURE_COLUMNS].mean().reset_index()
    # president_mean_vectors.to_csv("presidents_vectors.csv", index=False)
    df_filtered = df.copy()

    # === Step 1: Feature extraction ===
    df_filtered = add_topic_columns(df_filtered)
    df_filtered = add_emotion_columns(df_filtered)
    df_filtered = add_label_columns(df_filtered)

    # === Step 2: Compute speech length for weighting ===
    df_filtered['length'] = df_filtered['speech'].apply(lambda x: len(str(x).split()))


    clustered_rows = []

    for president, group in df_filtered.groupby('President'):
        speeches = group[FEATURE_COLUMNS].values
        weights = group['length'].values

        if len(group) >= 20:
            # === Case 1: Use KMeans to cluster speeches ===
            kmeans = KMeans(n_clusters=20, random_state=42)
            cluster_labels = kmeans.fit_predict(speeches)
        else:
            # === Case 2: Assign each speech to its own cluster ===
            cluster_labels = np.arange(len(group))

        group = group.copy()
        group['cluster'] = cluster_labels

        for cluster_id in sorted(group['cluster'].unique()):
            cluster_data = group[group['cluster'] == cluster_id]
            cluster_vector = np.average(cluster_data[FEATURE_COLUMNS], axis=0, weights=cluster_data['length'])

            clustered_rows.append({
                'President': president,
                'Cluster': cluster_id,
                **{col: val for col, val in zip(FEATURE_COLUMNS, cluster_vector)}
            })

    # === Step 4: Save the clustered vectors ===
    df_pres = pd.DataFrame(clustered_rows)
    df_pres.to_csv("presidents_vectors.csv", index=False)

    # === Step 5: Save the full speech-level vectors too ===
    df_filtered.to_excel("database_vectors_president.xlsx", index=False)

def president_predict(text : str):
    # Maybe Preprocess
    global count, df_president
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

    csv_vectors = df_president[FEATURE_COLUMNS].values

    # Step 4: Convert your input vector to correct shape
    vector = np.array(input_vector).reshape(1, -1)

    # Step 5: Compute cosine similarities
    similarities = cosine_similarity(vector, csv_vectors)[0]

    # Step 6: Find the index of the most similar vector
    most_similar_index = np.argmax(similarities)

    # Step 7: Get the most similar row
    most_similar_row = df_president.iloc[most_similar_index]

    # Step 8: Print or use the result
    count += 1
    if(count % 40 == 0):
        print(count)
    return most_similar_row["President"]

def test_loss(input_vector):
    input_vector = input_vector[FEATURE_COLUMNS]
    csv_vectors = df_president[FEATURE_COLUMNS].values

    # Step 4: Convert your input vector to correct shape
    vector = np.array(input_vector).reshape(1, -1)

    # Step 5: Compute cosine similarities
    similarities = cosine_similarity(vector, csv_vectors)[0]

    # Step 6: Find the index of the most similar vector
    most_similar_index = np.argmax(similarities)

    # Step 7: Get the most similar row
    most_similar_row = df_president.iloc[most_similar_index]

    return most_similar_row["President"]


def misclassification_loss(df):
    # Ensure columns exist
    if 'President' not in df.columns or 'predicted_president' not in df.columns:
        raise ValueError("DataFrame must contain 'President' and 'predicted_president' columns")

    # Count mismatches
    df_valid = df[df['predicted_president'].notna() & (df['predicted_president'].astype(str).str.strip() != '')]

    incorrect = (df_valid['President'] != df_valid['predicted_president']).sum()
    total = len(df_valid)
    print('Incorrect predictions: ', incorrect)
    print('Incorrect predictions \ total: ', incorrect / total)

df = pd.read_excel('combined_predictions.xlsx')
df = df[df['topics'] != 'None']
df = df[df['predicted_emotion'] != 'neutral']
create_presidents_vectors(df)
df = pd.read_excel("database_vectors_president.xlsx")
df['predicted_president'] = df.apply(test_loss, axis=1)
misclassification_loss(df)
df.to_excel("president_prediction_result.xlsx", index=False)