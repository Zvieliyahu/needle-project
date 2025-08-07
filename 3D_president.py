import pandas as pd
import streamlit as st
from predictor_helper import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
def create_presidents_vectors(df):
    df_filtered = df.copy()
    # Create vectors for each row - same as party
    df_filtered = add_topic_columns(df_filtered)
    df_filtered = add_emotion_columns(df_filtered)
    df_filtered = add_label_columns(df_filtered)
    president_mean_vectors = df_filtered.groupby('President')[FEATURE_COLUMNS].mean().reset_index()

    # Extract the Party for each President (assumes each President has one unique Party)
    president_parties = df_filtered[['President', 'Party']].drop_duplicates(subset='President')

    # Merge Party info back into the mean vectors
    president_mean_vectors = president_mean_vectors.merge(president_parties, on='President', how='left')
    pd.set_option("display.max_columns", None)
    print(president_mean_vectors.head())
    president_mean_vectors.to_csv("mean_presidents_vectors.csv", index=False)

###############################################################################################################
# All parties
#
st.set_page_config(layout="wide")
st.title("3D Visualization of U.S. Presidents Based on Topics & Emotions")
#
# # Sample data — replace with your actual CSV or source
#
# #df = pd.DataFrame(data)
df = pd.read_csv("mean_presidents_vectors.csv")

# Extract features: exclude 'President' and 'Party'
features = df.drop(columns=['President', 'Party'])
presidents = df['President']
parties = df['Party']

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Apply PCA to reduce to 3D
pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

# Create DataFrame for visualization
df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])
df_pca['President'] = presidents
df_pca['Party'] = parties
party_colors = {
    'Democratic': 'blue',
    'Republican': 'red',
    'Democratic-Republican' : 'green',
    'National Union': 'orange',   # example for a third party
    'Whig': 'grey',             # optional
    'Unaffiliated' : 'teal',
    'Federalist': 'pink'        # optional
}

# Create interactive 3D scatter plot
fig = px.scatter_3d(
    df_pca,
    x='PC1',
    y='PC2',
    z='PC3',
    color='Party',
    color_discrete_map=party_colors,  # ✅ force consistent colors
    text='President',
    title='3D PCA of Presidents: Topics and Emotions by Party',
    height=700
)

fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)
###################################################################################################################
# Only democratic and republican
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


########################################################################################################################
# Preparation
# df = pd.read_excel('combined_predictions.xlsx')
# df = df[df['topics'] != 'None']
# df = df[df['predicted_emotion'] != 'neutral']
# create_presidents_vectors(df)