from pysentimiento import create_analyzer
from cleanData import clean_presidential_speeches
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

tqdm.pandas()


# Load the GoEmotions classifier
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-go-emotion",
    top_k=1,
    truncation=True
)


def classify_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifying each speech with emotion.
    :param df: data frame with the column speech
    :return: a new data frame with a predicted emotion column
    """
    df = df.copy()

    # Define classification logic
    def get_top_emotion(text):
        try:
            trimmed_text = ' '.join(text.split()[20:])  # trimming the thank you part
            result = emotion_classifier(text)
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
    result_df.to_excel("speeches_with_emotions_4.xlsx", index=False)
