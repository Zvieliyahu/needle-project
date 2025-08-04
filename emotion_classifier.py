from cleanData import clean_presidential_speeches
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import spacy

tqdm.pandas()

non_topic_words = [
    # Your previous words:
    "go", "think", "want", "know", "say", "good", "lot", "get", "like",
    "thank", "time", "people", "work", "today", "thing", "great", "come",
    "look", "way", "right", "help", "need", "make", "well", "let", "tell",
    "see", "take", "try", "keep", "talk", "ask", "use", "give", "put",
    "feel", "seem", "leave", "mean", "start", "call", "show", "really",
    "big", "year", "new", "last", "many", "still", "find", "even",

    # Additional filler/generic/common words often not topic-specific:
    "also", "much", "much", "very", "always", "never", "quite", "just",
    "lot", "some", "most", "more", "most", "better", "better", "much",
    "well", "today", "tonight", "okay", "ok", "yes", "no", "let's", "us",
    "got", "getting", "did", "didn't", "does", "doesn't", "do", "don't",
    "will", "would", "could", "should", "might", "must", "can", "shall",
    "probably", "maybe", "actually", "basically", "seriously", "honestly",
    "definitely", "literally", "sure", "right", "okay", "alright", "hey",
    "hello", "hi", "well", "oh", "um", "uh", "yeah", "okay", "anyway",
    "anyways", "however", "therefore", "thus", "meanwhile", "although",
    "though", "yet", "still", "actually", "simply", "quite", "pretty",
    "rather", "soon", "often", "always", "never", "sometimes", "usually",
    "again", "already", "soon", "later", "ago",

    # Common pronouns (sometimes too generic for topic detection):
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",

    # Common conjunctions and prepositions that spaCy usually removes but can add:
    "and", "or", "but", "if", "because", "while", "as", "until", "when",
    "where", "after", "before", "since", "though", "although", "nor",
    "so", "than", "whether",

    # Other words we've missed
    "president", "go", "united", "states", "america", "country", "say", "nation", "world", "american"
]


# Load the GoEmotions classifier
emotion_classifier = pipeline(
    "text-classification",
    model="bhadresh-savani/bert-base-go-emotion",
    top_k=1,
    truncation=True
)

stop_words = set(stopwords.words('english')) # | set(non_topic_words)
nlp = spacy.load("en_core_web_sm")
for word in non_topic_words:
    nlp.vocab[word].is_stop = True


def classify_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifying each speech with emotion.
    :param df: data frame with the column speech
    :return: a new data frame with a predicted emotion column
    """
    df = df.copy()

    # Define classification logic
    def get_top_emotion(text):
        doc = nlp(text.lower())

        tokens = [
            token.lemma_ for token in doc
            if token.is_alpha
               and token.pos_ == "ADJ"  # Only adjectives
               and token.lemma_.lower() not in stop_words
        ]

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
    result_df.to_excel("speeches_with_emotions_4.xlsx", index=False)
