import re

from cleanData import clean_presidential_speeches
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import nltk
nltk.download('stopwords')
import spacy

tqdm.pandas()


# Load the GoEmotions classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
    truncation=True
)

nlp = spacy.load("en_core_web_sm")
ALLOWED_POS = {"ADJ", "VERB", "NOUN", "ADV"}


def remove_thanking_phrases(text):
    """
    A helper function for classify emotion that removes all thanks (to avoid classifying it to gratitude)
    :param text: A string of the speech
    :return: cleaned version (removed thanks)
    """
    patterns = [
        r"\bthank(s| you| you all)?\b"
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.I)
    return text


def classify_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifying each speech with emotion.
    :param df: data frame with the column speech
    :return: a new data frame with a predicted emotion column
    """
    df = df.copy()

    # Define classification logic
    def get_top_emotion(text):
        # Converting text to a nlp object and filtering the tokens
        doc = nlp(remove_thanking_phrases(text).lower())

        tokens = [
            token.lemma_ for token in doc
            if token.is_alpha and token.pos_ in ALLOWED_POS
        ]

        # Classifying emotion or neutral if unsuccessful
        try:
            result = emotion_classifier(' '.join(tokens))
            return result[0][0]['label'] if result else "neutral"
        except Exception as e:
            print(f"Error processing text: {text[:30]}... -> {e}")
            return "error"

    # Apply classifier to each speech
    df['predicted_emotion'] = df['speech'].progress_apply(get_top_emotion)
    return df


if __name__ == "__main__":
    speeches_df = clean_presidential_speeches(r'Data\presidential_speeches.xlsx')
    result_df = classify_emotion(speeches_df)
    result_df.to_excel("speeches_with_emotions_final.xlsx", index=False)
