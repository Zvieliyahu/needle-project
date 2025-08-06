import pandas as pd
import numpy as np
from process_helper import *
from check_common_words import *
from process_data import *

# Adds a topic columns for the dataframe
def add_topic_columns(df):
    # Clean and normalize the 'topics' column
    df['topics'] = (
        df['topics']
        .fillna('')
        .str.replace(';', ',', regex=False)
        .str.lower()
    )

    # Add binary columns for each topic
    for topic in TOPICS:
        df[topic] = df['topics'].str.contains(topic).astype(int)
    return df

# Adds a emotions columns for the dataframe
def add_emotion_columns(df):
    # Clean the 'predicted_emotion' column (ensure strings and no NaNs)
    df['predicted_emotion'] = df['predicted_emotion'].fillna('').astype(str).str.strip().str.lower()
    for emotion in EMOTIONS:
        df[emotion] = (df['predicted_emotion'] == emotion).astype(int)
    return df

# Adds a label columns for the dataframe
def add_label_columns(df):
    # Clean the 'label' column (ensure strings, no NaNs, uppercase)
    df['label'] = df['label'].fillna('').astype(str).str.upper()
    for label in LABELS:
        df[label] = (df['label'] == label).astype(int)
    return df

