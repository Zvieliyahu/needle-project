from process_data import *
# TODO: Check
# from filter_topic import assign_positivity_label
from tqdm import tqdm
WINDOW_SIZE = 15
MAX_SAMPLES_PER_TOPIC = 10


def add_topic_columns(df):
    """
    Adds topic columns for the dataframe - column for each valid topic (1 if exists, 0 if not).
    :param df: Dataframe with a topic column.
    :return: Updated dataframe.
    """
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


def add_emotion_columns(df):
    """
    Adds emotion columns for the dataframe - column for each valid emotion (1 if exists, 0 if not).
    :param df: Dataframe with an emotion column.
    :return: Updated dataframe.
    """
    df['predicted_emotion'] = df['predicted_emotion'].fillna('').astype(str).str.strip().str.lower()
    for emotion in EMOTIONS:
        df[emotion] = (df['predicted_emotion'] == emotion).astype(int)
    return df


def add_label_columns(df):
    """
    Adds label columns for the dataframe - column for each valid label - sentiment (1 if exists, 0 if not).
    :param df: Dataframe with a label column.
    :return: Updated dataframe.
    """
    df['label'] = df['label'].fillna('').astype(str).str.upper()
    for label in LABELS:
        df[label] = (df['label'] == label).astype(int)
    return df


def add_topic_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds topic sentiment columns for the dataframe - column for each valid topic.
    :param df: Dataframe with a topic column.
    :return: Updated dataframe.
    """
    df = df.copy()

    def process_speech(speech: str, topic_string: str) -> dict:
        """
        Calculates speech topic's sentiment.
        :param speech: Speech to be analyzed.
        :param topic_string: Topics to be analyzed.
        :return: Relevant topics sentiment.
        """
        relevant_topics = [
            t.strip() for t in topic_string.split(",")
            if t.strip() and t.strip().lower() != "none"
        ]

        speech = remove_thanking_phrases(speech)
        words = speech.split()
        lower_words = [w.lower() for w in words]
        topic_scores = {}

        # For each relevant topic calculate sentiment (1 - very positive, -1 - very negative, 0 - natural)
        for topic in relevant_topics:
            topic_clean = topic.strip().title()
            keywords = TOPICS_FOR_CLASSIFICATION.get(topic_clean, [])
            if not keywords:
                topic_scores[topic] = 0.0
                continue

            indices = []
            for i, word in enumerate(lower_words):
                word_clean = re.sub(r'\W+', '', word)
                if word_clean in keywords:
                    indices.append(i)

            if not indices:
                topic_scores[topic] = 0.0
                continue

            step = max(1, len(indices) // MAX_SAMPLES_PER_TOPIC)
            sampled_indices = indices[:MAX_SAMPLES_PER_TOPIC * step:step]

            score = 0
            for idx in sampled_indices:
                start = max(0, idx - WINDOW_SIZE)
                end = min(len(words), idx + WINDOW_SIZE + 1)
                context = " ".join(words[start:end])
                result = assign_positivity_label(context)
                label = result["label"]

                if label == "positive":
                    score += 1
                elif label == "negative":
                    score -= 1

            normalized_score = score / len(sampled_indices)
            topic_scores[topic] = round(normalized_score, 4)

        return topic_scores

    tqdm.pandas(desc="Processing speeches")
    df["__temp_topic_scores__"] = df.progress_apply(
        lambda row: process_speech(row["speech"], row["topics"]), axis=1
    )
    # Add columns of topic sentiment - 0 if topic does not appear
    for topic in TOPICS_FOR_CLASSIFICATION:
        column_name = f"{topic.lower()}_sentiment"
        df[column_name] = df["__temp_topic_scores__"].apply(lambda d: d.get(topic.lower(), 0.0))
    df.drop(columns=["__temp_topic_scores__"], inplace=True)
    return df
