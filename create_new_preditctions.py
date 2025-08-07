from filter_topic import *

if __name__ == '__main__':
    original_speeches_df = clean_presidential_speeches('Data/presidential_speeches.xlsx')
    # Adding sentiment and emotion: #
    tqdm.pandas()
    sentiment_results = original_speeches_df['speech'].progress_apply(assign_positivity_label)
    # Convert the series of dicts into a DataFrame with separate columns
    sentiment_df = sentiment_results.apply(pd.Series)
    # Join the new sentiment columns to your original DataFrame
    topic_speeches_df = pd.concat([original_speeches_df, sentiment_df], axis=1)

    topic_speeches_df = classify_emotion(topic_speeches_df)

    # Saving filtered data frame
    topic_speeches_df.to_excel(f"second predictions/emotion_and_positivity_predictions.xlsx", index=False)
