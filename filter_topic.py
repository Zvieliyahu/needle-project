from typing import List, Dict
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
from filter_topic_helper import *
from cleanData import clean_presidential_speeches
from process_data import remove_thanking_phrases

def classify_emotion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifying each speech with emotion using chunking.
    :param df: DataFrame with a 'speech' column
    :return: DataFrame with added 'predicted_emotion' column
    """
    df = df.copy()

    def get_top_emotion(text):
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


def assign_positivity_label(speech: str) -> Dict:
    """
    Assign positivity label (positive, negative or neutral) to a given text.
    :param speech: the text to which to assign the label
    :return: a dict of label: confidence
    """
    speech = remove_thanking_phrases(speech)
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

    if not labels:
        return {
            "label": "unknown",
            "confidence": -1
        }

    majority_label = Counter(labels).most_common(1)[0][0]
    avg_score = sum(scores) / len(scores)

    return {
        "label": majority_label,
        "confidence": round(avg_score, 4)
    }


def detect_words(text: str, keywords: List[str], important_keywords: List[str], min_appearances=0):
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

    # Find all match positions of important keywords using regex
    match_spans = []
    for kw in important_keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        for match in re.finditer(pattern, text_lower):
            match_spans.append((match.start(), match.end()))

    if not match_spans:
        print("ERROR: Failed to find at least one keyword.")
        return ""

    # Determine cut region by character positions
    first_match_start = min(start for start, _ in match_spans)
    last_match_end = max(end for _, end in match_spans)

    # Convert character range to word-based range for padding
    words = re.findall(r'\b\w+\b', text_lower)
    word_starts = [m.start() for m in re.finditer(r'\b\w+\b', text_lower)]

    # Find word indices closest to match boundaries
    first_word_index = next((i for i, pos in enumerate(word_starts) if pos >= first_match_start), 0)
    last_word_index = next((i for i, pos in enumerate(word_starts) if pos >= last_match_end), len(words) - 1)

    # Add padding
    first_word_index = max(0, first_word_index - num_of_extra_words)
    last_word_index = min(len(words) - 1, last_word_index + num_of_extra_words)

    selected_words = words[first_word_index:last_word_index + 1]
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
    speeches_df = clean_presidential_speeches(file_path)

    results = speeches_df['speech'].apply(lambda x: detect_words(x, keywords, important_keywords, appearance_threshold))

    # Assign results back to DataFrame columns
    speeches_df[[f'contains_{topic}_keywords', 'words_found']] = pd.DataFrame(results.tolist(),
                                                                              index=speeches_df.index)

    # Filter and export whole speeches
    topic_speeches_df = speeches_df[speeches_df[f'contains_{topic}_keywords'] == True]
    topic_speeches_df.to_excel(f"{directory}whole_speeches_that_include_{topic}_keywords.xlsx", index=False)

    # Filtering original speeches
    # original_speeches_df = pd.read_excel('Data/presidential_speeches.xlsx')
    # filtered_original_df = original_speeches_df.loc[topic_speeches_df.index]

    # Cutting speech for relevant part only
    topic_speeches_df['speech'] = topic_speeches_df['speech'].apply(
        lambda x: cut_speech(x, important_keywords, num_of_extra_words=60)
    )
    topic_speeches_df['topic'] = topic

    # Adding sentiment and emotion: #
    tqdm.pandas()
    sentiment_results = topic_speeches_df['speech'].progress_apply(assign_positivity_label)
    # Convert the series of dicts into a DataFrame with separate columns
    sentiment_df = sentiment_results.apply(pd.Series)
    # Join the new sentiment columns to your original DataFrame
    topic_speeches_df = pd.concat([topic_speeches_df, sentiment_df], axis=1)

    topic_speeches_df = classify_emotion(topic_speeches_df)

    # Saving filtered data frame
    topic_speeches_df.to_excel(f"{directory}cutted_speeches_that_include_{topic}_keywords.xlsx", index=False)
    return topic_speeches_df.copy()


if __name__ == '__main__':
    # find_speeches_with_keywords(FILE_PATH_IMMIGRATION, IMMIGRATION_KEYWORDS, MOST_IMPORTANT_KEYWORDS_IMMIGRATION,
    #                             MIN_APPEARANCES, DIRECTORY_IMMIGRATION, "immigration")
    # find_speeches_with_keywords(FILE_PATH_BLACK_RIGHTS, BLACK_RIGHTS_KEYWORD, MOST_IMPORTANT_BLACK_RIGHTS_KEYWORDS,
    #                             MIN_APPEARANCES, DIRECTORY_BLACK_RIGHTS, "black_rights")
    # find_speeches_with_keywords(FILE_PATH, WOMEN_RIGHTS_KEYWORDS, MOST_IMPORTANT_KEYWORDS_WOMEN_RIGHTS,
    #                             MIN_APPEARANCES, DIRECTORY_WOMEN_RIGHTS, "womens_rights")
    find_speeches_with_keywords(FILE_PATH, NATIVE_AMERICANS_KEYWORDS, MOST_IMPORTANT_KEYWORDS_NATIVE_AMERICANS,
                                MIN_APPEARANCES, DIRECTORY_NATIVE_AMERICANS, "native_americans")
