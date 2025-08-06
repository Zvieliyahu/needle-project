
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