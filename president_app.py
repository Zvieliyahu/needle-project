
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
from president_predictor import president_predict  # import your function

st.set_page_config(layout="wide")

st.title("Political President Predictor")
st.write("Enter a political speech or text below to predict to which **president** it aligns the most.")

text_input = st.text_area("Input speech text here:", height=200, key="input_text_area")

# Load data once
df = pd.read_csv("mean_presidents_vectors.csv")

features = df.drop(columns=['President', 'Party'])
presidents = df['President']
parties = df['Party']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

pca = PCA(n_components=3)
pca_result = pca.fit_transform(X_scaled)

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
    'Federalist': 'pink',        # optional
    'Input': 'black',  # Color for the input vector point
}

if st.button("Predict", key="predict_button"):
    if text_input.strip():
        with st.spinner("Analyzing..."):
            input_vector, president = president_predict(
                text_input)  # your function returns the vector (same dim as features) and predicted president

            st.success(f"Predicted President: **{president}**")

            # Standardize the input vector using the fitted scaler
            input_scaled = scaler.transform(np.array(input_vector).reshape(1, -1))
            # Apply PCA transform to the input
            input_pca = pca.transform(input_scaled)

            # Append the input point to df_pca
            input_row = pd.DataFrame({
                'PC1': [input_pca[0, 0]],
                'PC2': [input_pca[0, 1]],
                'PC3': [input_pca[0, 2]],
                'President': ['Input Text'],
                'Party': ['Input']
            })
            df_pca_with_input = pd.concat([df_pca, input_row], ignore_index=True)

            # Plot with the input point included
            fig = px.scatter_3d(
                df_pca_with_input,
                x='PC1',
                y='PC2',
                z='PC3',
                color='Party',
                color_discrete_map=party_colors,
                text='President',
                title='3D PCA of Presidents: Topics and Emotions with Input',
                height=700
            )
            fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please enter some text to analyze.")
else:
    # Show the base plot without input if no prediction yet
    fig = px.scatter_3d(
        df_pca,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Party',
        color_discrete_map=party_colors,
        text='President',
        title='3D PCA of Presidents: Topics and Emotions',
        height=700
    )
    fig.update_traces(marker=dict(size=6), selector=dict(mode='markers'))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    st.plotly_chart(fig, use_container_width=True)