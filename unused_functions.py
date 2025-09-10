
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



def extract_sentiments(speech: str) -> float:
    """
    Extract a sentiment from a speech, score > 0 is positive, score < 0 is negative.
    :param speech: string of speech
    :return: positivity score
    """
    blob = TextBlob(remove_thanking_phrases(speech))
    sentiment = blob.sentiment.polarity
    # possibly process more
    return sentiment

# df = add_topic_columns(df)
# df = add_emotion_columns(df)
# df = add_label_columns(df)
# X = df[FEATURE_COLUMNS]
# y = df['Party']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#
# logreg_model = LogisticRegression(max_iter=1000, random_state=42)
# logreg_model.fit(X_train, y_train)
#
# # Optional: check performance
# y_pred = logreg_model.predict(X_test)
# print(classification_report(y_test, y_pred))
#
# # === Step 3: Use the trained model to predict ===
# df = clean_presidential_speeches('Data\presidential_speeches.xlsx')
# df['predicted_party'] = df['speech'].head(100).apply(lambda text: predict_party(text, model=logreg_model))
#
# misclassification_loss(df)
# # df['predicted_party'] = df['speech'].apply(predict_party)
# # misclassification_loss(df)
# df.to_excel("log_prediction_result.xlsx", index=False)
# df = pd.read_excel("emotions_filtered_by_positivity_label.xlsx")
# df = df[df['Party'].isin(['Democratic', 'Republican'])]
# df = add_topic_columns(df)
# df = add_emotion_columns(df)
# df = add_label_columns(df)
#
# X = df[FEATURE_COLUMNS]
# y = df['Party']
#
# # === Step 1: K-Fold Cross-Validation to evaluate model ===
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# logreg = LogisticRegression(max_iter=1000, random_state=42)
#
# cv_scores = cross_val_score(logreg, X, y, cv=kf, scoring='accuracy')
# print(f"Cross-validation accuracies: {cv_scores}")
# print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
#
# # === Step 2: Train-Test split and final training ===
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
# logreg_model = LogisticRegression(max_iter=1000, random_state=42)
# logreg_model.fit(X_train, y_train)
#
# # Evaluate on test set
# y_pred = logreg_model.predict(X_test)
# print("Test set classification report:")
# print(classification_report(y_test, y_pred))
#
# # === Step 3: Use the trained model for predictions on new speeches ===
# df_new = clean_presidential_speeches('Data/presidential_speeches.xlsx')
#
# # Only predict on first 100 speeches as you had it
# df_new['predicted_party'] = df_new['speech'].head(100).apply(lambda text: predict_party(text, model=logreg_model))
#
# misclassification_loss(df_new)
#
# df_new.to_excel("log_prediction_result.xlsx", index=False)

def clean_text(text: str):
    """
    Cleaning text by removing stopwords and non alphabet characters.
    :param text: a speech
    :return:
    """
    doc = nlp(text.lower())

    # Filter: remove stopwords, punctuation, numbers, and non-alphabetic tokens
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and token.lemma_.lower() not in stop_words
    ]

    return ' '.join(tokens)



def plot_emotions_and_labels(df, start_year=None, end_year=None):
    """
    Plots a grouped bar chart of predicted emotions and labels by topic over a specified period.

    Parameters:
        df (DataFrame): Must contain 'date', 'topic', 'predicted_emotion', and 'label' columns.
        start_year (int, optional): Start of the period (inclusive).
        end_year (int, optional): End of the period (inclusive).
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Filter by year range
    if start_year is not None:
        df = df[df['date'].dt.year >= start_year]
    if end_year is not None:
        df = df[df['date'].dt.year <= end_year]

    # Group and count emotions
    emotions_grouped = df.groupby(['topic', 'predicted_emotion']).size().reset_index(name='count')
    emotions_pivot = emotions_grouped.pivot(index='topic', columns='predicted_emotion', values='count').fillna(0)

    # Group and count labels
    labels_grouped = df.groupby(['topic', 'label']).size().reset_index(name='count')
    labels_pivot = labels_grouped.pivot(index='topic', columns='label', values='count').fillna(0)

    topics = emotions_pivot.index
    emotions = emotions_pivot.columns
    labels = labels_pivot.columns

    width = 0.35  # width of each bar
    x = np.arange(len(topics))  # the label locations

    fig, ax = plt.subplots(figsize=(14,6))

    # Plot emotions bars
    for i, emotion in enumerate(emotions):
        ax.bar(x - width/2 + (i/len(emotions))*width, emotions_pivot[emotion], width/len(emotions), label=f"Emotion: {emotion}")

    # Plot labels bars
    for i, lbl in enumerate(labels):
        ax.bar(x + width/2 + (i/len(labels))*width, labels_pivot[lbl], width/len(labels), label=f"Label: {lbl}", hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(topics, rotation=45, ha='right')
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Speeches")
    ax.set_title(f"Speeches by Emotion and Label per Topic ({start_year}-{end_year})")
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.show()
