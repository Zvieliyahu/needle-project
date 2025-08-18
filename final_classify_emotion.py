import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
from filter_topic_helper import *
from process_helper import *
from process_data import remove_thanking_phrases

nlp = spacy.load("en_core_web_sm")

def classify_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifying each speech with emotion using chunking and preprocessing.
    :param df: DataFrame with a 'speech' column
    :return: DataFrame with added 'predicted_emotion' column
    """
    df = df.copy()
    # print("IN EMOTION CLASSIFICATION !!!!!!!!!!!!!!!!!!!!!!!")
    def get_top_emotion(text):
        doc = nlp(remove_thanking_phrases(text).lower())

        tokens = [
            token.lemma_ for token in doc
            if token.is_alpha and token.pos_ in ALLOWED_POS
        ]
        try:
            words = text.split()
            chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

            emotion_labels = []

            for chunk in chunks:
                result = emotion_classifier(chunk, truncation=True)
                label = result[0][0]['label'] if result else "neutral"
                emotion_labels.append(label)

            # Majority label across chunks
            if emotion_labels:
                majority_emotion = Counter(emotion_labels).most_common(1)[0][0]
                return majority_emotion
            else:
                return "neutral"

        except Exception as e:
            print(f"Error processing text: {text[:30]}... -> {e}")
            return "error"

    tqdm.pandas()
    df['predicted_emotion'] = df['speech'].progress_apply(get_top_emotion)
    return df
