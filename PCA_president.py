from predictor_helper import *
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


def create_presidents_vectors(df: pd.DataFrame):
    """
    Creates the vectors describing each president in the dataset.
    :param df: Dataframe with processed speeches of presidents.
    :return:
    """
    df_filtered = df.copy()
    # Create vectors for each row - same as party (without topic specific sentiment)
    df_filtered = add_topic_columns(df_filtered)
    df_filtered = add_emotion_columns(df_filtered)
    df_filtered = add_label_columns(df_filtered)
    president_mean_vectors = df_filtered.groupby('President')[FEATURE_COLUMNS_3D].mean().reset_index()

    # Extract the Party and years of presidency for each President (assumes each President has one unique values)
    president_parties = df_filtered[['President', 'Party', 'from', 'until']].drop_duplicates(subset='President')
    # Merge additional info back into the mean vectors
    president_mean_vectors = president_mean_vectors.merge(president_parties, on='President', how='left')
    president_mean_vectors.to_csv("mean_presidents_vectors.csv", index=False)


if __name__ == '__main__':
    """
    Streamlit application of visualization of the top 10 presidents vectors (Democratic and Republican)
    """
    st.set_page_config(layout="wide")
    st.title("1D Visualization of U.S. Presidents (Democratic and Republican) Based on Topics, Emotions & Sentiments")

    # Load data
    df = pd.read_csv("mean_presidents_vectors.csv")
    df = df[df['Party'].isin(['Democratic', 'Republican'])]
    features = df.drop(columns=['President', 'Party'])
    presidents = df['President']
    parties = df['Party']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA to 1D
    pca = PCA(n_components=1)
    pca_result = pca.fit_transform(X_scaled)

    # PCA dataframe and reverse axis - for visualization
    df_pca = pd.DataFrame(data=pca_result * -1, columns=['PC1'])
    df_pca['President'] = presidents
    df_pca['Party'] = parties

    # Add stats (original features)
    df_pca = pd.concat([df_pca, features.reset_index(drop=True)], axis=1)

    # Sort by PC1 for correct left-right ordering
    df_pca = df_pca.sort_values(by='PC1').reset_index(drop=True)

    # y=0 for all dots
    df_pca['y'] = 0

    # Alternate label positions - for visualization
    text_positions = ['top center' if i % 2 == 0 else 'bottom center' for i in range(len(df_pca))]
    for i, name in enumerate(df_pca['President']):
        if 'George W. Bush' in name:
            text_positions[i] = 'bottom center'

    # Party color map
    party_colors = {
        'Democratic': 'blue',
        'Republican': 'red',
    }
    df_pca['text_label'] = df_pca['President'].str.replace(r'(\.)', r'\1<br>', regex=True) + '<br>' + df_pca[
        'from'].astype(str) + '-' + df_pca['until'].astype(str)

    # Plot
    fig = go.Figure()

    for party in df_pca['Party'].unique():
        df_party = df_pca[df_pca['Party'] == party]

        fig.add_trace(go.Scatter(
            x=df_party['PC1'],
            y=df_party['y'],
            mode='markers+text',
            name=party,
            marker=dict(size=10, color=party_colors[party]),
            text=df_party['text_label'],
            textposition=[text_positions[i] for i in df_party.index],
            textfont=dict(size=16),
        ))

    fig.update_layout(
        title='1D PCA of Presidents',
        height=700,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title='PC1',
        yaxis=dict(visible=False),
        legend=dict(title='Party'),
    )

    # Display
    st.plotly_chart(fig, use_container_width=True)
