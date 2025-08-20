import pandas as pd
import streamlit as st
from predictor_helper import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
def create_presidents_vectors(df):
    df_filtered = df.copy()
    # Create vectors for each row - same as party
    df_filtered = add_topic_columns(df_filtered)
    df_filtered = add_emotion_columns(df_filtered)
    df_filtered = add_label_columns(df_filtered)
    president_mean_vectors = df_filtered.groupby('President')[FEATURE_COLUMNS_3D].mean().reset_index()
    # Extract the Party for each President (assumes each President has one unique Party)
    president_parties = df_filtered[['President', 'Party']].drop_duplicates(subset='President')

    # Merge Party info back into the mean vectors
    president_mean_vectors = president_mean_vectors.merge(president_parties, on='President', how='left')
    pd.set_option("display.max_columns", None)
    print(president_mean_vectors.head())
    president_mean_vectors.to_csv("mean_presidents_vectors.csv", index=False)

###############################################################################################################
# # All parties
# #
# st.set_page_config(layout="wide")
# st.title("3D Visualization of U.S. Presidents Based on Topics & Emotions")
# #
# # # Sample data — replace with your actual CSV or source
# #
# # #df = pd.DataFrame(data)
# df = pd.read_csv("mean_presidents_vectors.csv")
#
# # Extract features: exclude 'President' and 'Party'
# features = df.drop(columns=['President', 'Party'])
# presidents = df['President']
# parties = df['Party']
#
# # Standardize the feature values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(features)
#
# # Apply PCA to reduce to 3D
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(X_scaled)
#
# # Create DataFrame for visualization
# df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
# df_pca['President'] = presidents
# df_pca['Party'] = parties
# party_colors = {
#     'Democratic': 'blue',
#     'Republican': 'red',
#     'Democratic-Republican' : 'green',
#     'National Union': 'orange',   # example for a third party
#     'Whig': 'grey',             # optional
#     'Unaffiliated' : 'teal',
#     'Federalist': 'pink'        # optional
# }
#
# # Create interactive 3D scatter plot
# fig = px.scatter_3d(
#     df_pca,
#     x='PC1',
#     y='PC2',
#     z='PC3',
#     color='Party',
#     color_discrete_map=party_colors,  # ✅ force consistent colors
#     text='President',
#     title='3D PCA of Presidents: Topics and Emotions by Party',
#     height=700
# )
#
# fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
#
# # Display the plot in Streamlit
# st.plotly_chart(fig, use_container_width=True)
###################################################################################################################
# Only democratic and republican 3D
# st.set_page_config(layout="wide")
# st.title("3D Visualization of U.S. Presidents Based on Topics & Emotions")
#
# # Sample data — replace with your actual CSV or source
#
# #df = pd.DataFrame(data)
# df = pd.read_csv("mean_presidents_vectors.csv")
# df = df[df['Party'].isin(['Democratic', 'Republican'])]
# # df = df[df['President'] != 'Harry S. Truman']
# # df = df[df['President'] != 'Grover Cleveland']
#
#
#
# # Extract features: exclude 'President' and 'Party'
# features = df.drop(columns=['President', 'Party'])
# presidents = df['President']
# parties = df['Party']
#
# # Standardize the feature values
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(features)
#
# # Apply PCA to reduce to 3D
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(X_scaled)
#
# # Create DataFrame for visualization
# df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
# df_pca['President'] = presidents
# df_pca['Party'] = parties
# party_colors = {
#     'Democratic': 'blue',
#     'Republican': 'red',
# }
#
# # Create interactive 3D scatter plot
# fig = px.scatter_3d(
#     df_pca,
#     x='PC1',
#     y='PC2',
#     z='PC3',
#     color='Party',
#     color_discrete_map=party_colors,  # ✅ force consistent colors
#     text='President',
#     title='3D PCA of Presidents: Topics and Emotions by Party',
#     height=700
# )
#
# fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
#
# # Display the plot in Streamlit
# st.plotly_chart(fig, use_container_width=True)

######################################################################################################################
# Streamlit settings
st.set_page_config(layout="wide")
st.title("1D Visualization of U.S. Presidents Based on Topics, Emotions & Sentiments")

# Load data
df = pd.read_csv("mean_presidents_vectors.csv")
df = df[df['Party'].isin(['Democratic', 'Republican'])]

# Features
features = df.drop(columns=['President', 'Party'])
presidents = df['President']
parties = df['Party']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# PCA to 1D
pca = PCA(n_components=1)
pca_result = pca.fit_transform(X_scaled)

# PCA dataframe
df_pca = pd.DataFrame(data=pca_result * -1, columns=['PC1'])  # Reverse axis
df_pca['President'] = presidents
df_pca['Party'] = parties

# Add stats (original features)
df_pca = pd.concat([df_pca, features.reset_index(drop=True)], axis=1)

# Sort by PC1 for correct left-right ordering
df_pca = df_pca.sort_values(by='PC1').reset_index(drop=True)

# y=0 for all dots
df_pca['y'] = 0

# Alternate label positions
text_positions = ['top center' if i % 2 == 0 else 'bottom center' for i in range(len(df_pca))]
for i, name in enumerate(df_pca['President']):
    if 'George W. Bush' in name:
        text_positions[i] = 'bottom center'

# Party color map
party_colors = {
    'Democratic': 'blue',
    'Republican': 'red',
}

# Create plot
fig = go.Figure()

# Add points grouped by party for legend
for party in df_pca['Party'].unique():
    df_party = df_pca[df_pca['Party'] == party]

    fig.add_trace(go.Scatter(
        x=df_party['PC1'],
        y=df_party['y'],
        mode='markers+text',
        name=party,
        marker=dict(size=10, color=party_colors[party]),
        text=df_party['President'],
        textposition=[text_positions[i] for i in df_party.index],
        textfont=dict(size=12),
    ))

# Layout cleanup
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


########################################################################################################################
# Preparation
# df = pd.read_excel('final_data_predictions.xlsx')
# df_filtered_parties = df[df['Party'].isin(['Democratic', 'Republican'])]
#
# # Step 2: Count number of rows per president
# president_counts = df_filtered_parties['President'].value_counts()
#
# # Step 3: Get top 10 presidents by row count
# top_10_presidents = president_counts.head(10).index
#
# # Step 4: Filter the DataFrame to keep only those presidents
# df = df_filtered_parties[df_filtered_parties['President'].isin(top_10_presidents)]
# df = df[df['topics'] != 'None']
# df = df[df['predicted_emotion'] != 'neutral']
# create_presidents_vectors(df)
# Apply function to each speech
# df["__temp_topic_scores__"] = df["speech"].apply(lambda x: extract_topic_sentiment_scores(x, TOPICS_FOR_CLASSIFICATION))
#
# # For each topic, create a new column and extract score (default to 0)
# for topic in TOPICS_FOR_CLASSIFICATION:
#     column_name = f"{topic.lower()}_sentiment"
#     df[column_name] = df["__temp_topic_scores__"].apply(lambda d: d.get(topic, 0))
#
# # Optional: remove temporary column
# df.drop(columns=["__temp_topic_scores__"], inplace=True)