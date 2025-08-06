from typing import List, Dict
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
from filter_topic_helper import *


def classify_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifying each speech with emotion using chunking.
    :param df: DataFrame with a 'speech' column
    :return: DataFrame with added 'predicted_emotion' column
    """
    df = df.copy()

    def get_top_emotion(text):
        try:
            # Split into word chunks (no NLP filtering)
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


def assign_positivity_label(speech: str) -> Dict:
    """
    Assign positivity label (positive, negative or neutral) to a given text.
    :param speech: the text to which to assign the label
    :return: a dict of label: confidence
    """
    words = speech.split()
    chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

    labels = []
    scores = []

    for chunk in chunks:
        result_list = sentiment_pipeline(chunk, truncation=True)  # Always returns a list
        result = result_list[0][0]
        raw_label = result["label"]
        label = LABEL_MAP[raw_label]
        score = result["score"]

        if label == "neutral" and score < 0.6 and result_list[0][1]["score"] > 0.3:
            label = LABEL_MAP[result_list[0][1]["label"]]
            score = result_list[0][1]["score"]

        labels.append(label)
        scores.append(score)

    majority_label = Counter(labels).most_common(1)[0][0]
    avg_score = sum(scores) / len(scores)

    return {
        "label": majority_label,
        "positivity_score": round(avg_score, 4)
    }


def detect_words(text: List[str], keywords: List[str], important_keywords: List[str], min_appearances=0):
    """
    Detect given words in a speech.
    :param important_keywords: important keywords that must appear in the text
    :param min_appearances: a threshold of the amount of appearances of keywords in text
    :param text: The speech text
    :param keywords: A list of keywords to search for
    :return: a boolean if found keywords in speech and a dictionary of matched terms
    """
    text_lower = text.lower()
    found_terms = {}

    # Lowercase all keywords for comparison
    keywords_lower = [kw.lower() for kw in keywords]
    important_keywords_lower = [kw.lower() for kw in important_keywords]

    # Finding all matched words and summing their appearances in a dict
    for kw in keywords_lower:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            found_terms[kw] = len(matches)

    # Summing all appearances of words and verifying at least one important keyword was found
    num_appearances = 0
    include_important_keywords = False
    for term in found_terms:
        if term in important_keywords_lower:
            include_important_keywords = True
        num_appearances += found_terms[term]

    if found_terms and num_appearances >= min_appearances and include_important_keywords:
        found_keywords = ", ".join(f"{term}" for term in found_terms)
        return True, found_keywords
    return False, None


def cut_speech(text: str, important_keywords: List[str], num_of_extra_words=60):
    """
    Cutting a speech based on first and last appearance of a key word,with an extra words before and after
    the first and last words appear.
    :param text: a text to cut
    :param important_keywords:  keywords to look for in text
    :param num_of_extra_words: the "padding" of the text
    :return: a substring of the text sliced based on input or an empty string if did not find keywords
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    # Lowercase all keywords for comparison
    keywords_lower = [kw.lower() for kw in important_keywords]

    # Track positions of important keywords
    keyword_indices = [i for i, word in enumerate(words) if word.lower() in keywords_lower]

    if not keyword_indices:
        print("ERROR: Failed to find at least one keyword.")
        return ""  # No keyword found

    first_idx = max(keyword_indices[0] - num_of_extra_words, 0)
    last_idx = min(keyword_indices[-1] + num_of_extra_words + 1, len(words))

    # Slice the words and return the cut segment
    selected_words = words[first_idx:last_idx]
    return ' '.join(selected_words)


def find_speeches_with_keywords(file_path: str, keywords: List[str], important_keywords: List[str],
                                appearance_threshold: int, directory="", topic=""):
    """
    Finds speeches inside an Excel file based on the given input.
    :param file_path: a path to the Excel file
    :param keywords: keywords to look for in text
    :param important_keywords: list of keywords that at least one of them must appear in text
    :param appearance_threshold: the minimum number appearances of words from the keyword list to classify for the given
    topic
    :param directory: optional directory to save the output
    :param topic: the name of the topic
    :return: a data frame of the topic related parts of the texts that was classified to the given topic
    """
    speeches_df = pd.read_excel(file_path)

    results = speeches_df['speech'].apply(lambda x: detect_words(x, keywords, important_keywords, appearance_threshold))

    # Assign results back to DataFrame columns
    speeches_df[[f'contains_{topic}_keywords', 'words_found']] = pd.DataFrame(results.tolist(),
                                                                              index=speeches_df.index)

    # Filter and export whole speeches
    topic_speeches_df = speeches_df[speeches_df[f'contains_{topic}_keywords'] == True]
    topic_speeches_df.to_excel(f"{directory}whole_speeches_that_include_{topic}_keywords.xlsx", index=False)

    # Filtering original speeches
    original_speeches_df = pd.read_excel('Data/presidential_speeches.xlsx')
    filtered_original_df = original_speeches_df.loc[topic_speeches_df.index]

    # Cutting speech for relevant part only
    filtered_original_df['speech'] = filtered_original_df['speech'].apply(
        lambda x: cut_speech(x, important_keywords, num_of_extra_words=60)
    )
    filtered_original_df['topic'] = topic

    # Adding sentiment and emotion: #
    sentiment_results = filtered_original_df['speech'].apply(assign_positivity_label)
    # Convert the series of dicts into a DataFrame with separate columns
    tqdm.pandas()
    sentiment_df = sentiment_results.apply(pd.Series)
    # Join the new sentiment columns to your original DataFrame
    filtered_original_df = pd.concat([filtered_original_df, sentiment_df], axis=1)

    filtered_original_df = classify_emotion(filtered_original_df)

    # Saving filtered data frame
    filtered_original_df.to_excel(f"{directory}cutted_speeches_that_include_{topic}_keywords.xlsx", index=False)
    return filtered_original_df.copy()


if __name__ == '__main__':
    find_speeches_with_keywords(FILE_PATH, IMMIGRATION_KEYWORDS, MOST_IMPORTANT_KEYWORDS_IMMIGRATION,MIN_APPEARANCES, DIRECTORY, "immigration")
