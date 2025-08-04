
def extract_emotions(speech: str) -> dict:
    emotion = NRCLex(speech)
    print(emotion.raw_emotion_scores)


def extract_topic(speech: str) -> list:
    tokens = speech.split()  # tokenize by spaces
    matched_topics = {}
    for topic, keywords in topics.items():
        found_keywords = [kw for kw in keywords if kw in tokens]
        if found_keywords:
            matched_topics[topic] = {
                "count": len(found_keywords),
                "keywords": found_keywords
            }

    # Sort by number of matched keywords descending
    matched_topics = dict(sorted(matched_topics.items(), key=lambda x: x[1]["count"], reverse=True))
    filtered_topics = {topic: data for topic, data in matched_topics.items() if data["count"] > 2}
    filtered_topic_names = list(filtered_topics.keys())
    return filtered_topic_names
