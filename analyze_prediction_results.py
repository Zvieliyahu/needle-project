import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

POSITIVITY_THRESHOLD = 0.8
NEGATIVITY_THRESHOLD = 0.2


def plot_positivity_by_emotion(path: str, plot_filter: str):
    # Load the final merged DataFrame
    df = pd.read_excel(path)

    # Step 1: Filter rows where sentiment is strongly positive or negative

    # Step 2: Group by emotion and calculate average sentiment score and count
    avg_sentiment_per_emotion = (
        df
        .groupby('predicted_emotion')
        .agg(
            positivity_score=('positivity_score', 'mean'),
            count=('positivity_score', 'size')
        )
        .reset_index()
    )

    # Step 3: Assign sentiment labels for coloring only
    def assign_label(score):
        return 'POSITIVE' if score >= 0.5 else 'NEGATIVE'

    avg_sentiment_per_emotion['sentiment_label'] = avg_sentiment_per_emotion['positivity_score'].apply(assign_label)

    # Step 4: Assign bar colors based on sentiment label
    color_map = {'POSITIVE': 'blue', 'NEGATIVE': 'red'}
    bar_colors = avg_sentiment_per_emotion['sentiment_label'].map(color_map)

    # Step 5: Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        avg_sentiment_per_emotion['predicted_emotion'],
        avg_sentiment_per_emotion['positivity_score'],
        color=bar_colors
    )

    # Add average score and count on top of each bar (no sentiment label)
    for bar, count, avg_score in zip(
            bars,
            avg_sentiment_per_emotion['count'],
            avg_sentiment_per_emotion['positivity_score']
    ):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"Avg: {avg_score:.2f}\n(n={count})",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    # Aesthetics
    plt.title(f'Average Sentiment Score by Emotion (Strong Sentiments Only)\n'
              f'Used filter: {plot_filter}')
    plt.ylabel('Average Positivity Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add legend (colors only)
    legend_elements = [
        Patch(facecolor='blue', label='Positive'),
        Patch(facecolor='red', label='Negative')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title='Sentiment')

    # Save plot to file
    plt.savefig(f"emotions_filtered_by_{plot_filter}.png", dpi=300)
    plt.close()


def plot_positivity_by_emotion_with_thresholds(positivity_threshold=0.8, negativity_threshold=0.2):
    """
    Making a plot to see the correlation between emotions prediction and positivity
    :param positivity_threshold:
    :param negativity_threshold:
    :return:
    """
    # Load the final merged DataFrame
    df = pd.read_excel("combined_predictions.xlsx")

    # Step 1: Filter rows where sentiment is strongly positive or negative
    filtered_df = df[(df['positivity_score'] > positivity_threshold) | (df['positivity_score'] < negativity_threshold)]

    # Step 2: Group by emotion and calculate average sentiment score and count
    avg_sentiment_per_emotion = (
        filtered_df
        .groupby('predicted_emotion')
        .agg(
            positivity_score=('positivity_score', 'mean'),
            count=('positivity_score', 'size')
        )
        .reset_index()
    )

    # Step 3: Assign sentiment labels for coloring only
    def assign_label(score):
        return 'POSITIVE' if score >= 0.5 else 'NEGATIVE'

    avg_sentiment_per_emotion['sentiment_label'] = avg_sentiment_per_emotion['positivity_score'].apply(assign_label)

    # Step 4: Assign bar colors based on sentiment label
    color_map = {'POSITIVE': 'blue', 'NEGATIVE': 'red'}
    bar_colors = avg_sentiment_per_emotion['sentiment_label'].map(color_map)

    # Step 5: Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        avg_sentiment_per_emotion['predicted_emotion'],
        avg_sentiment_per_emotion['positivity_score'],
        color=bar_colors
    )

    # Add average score and count on top of each bar (no sentiment label)
    for bar, count, avg_score in zip(
            bars,
            avg_sentiment_per_emotion['count'],
            avg_sentiment_per_emotion['positivity_score']
    ):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"Avg: {avg_score:.2f}\n(n={count})",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

    # Aesthetics
    plt.title(f'Average Sentiment Score by Emotion (Strong Sentiments Only)\n'
              f'positivity threshold: {positivity_threshold}, negativity threshold: {negativity_threshold}')
    plt.ylabel('Average Positivity Score')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add legend (colors only)
    legend_elements = [
        Patch(facecolor='blue', label='Positive'),
        Patch(facecolor='red', label='Negative')
    ]
    plt.legend(handles=legend_elements, loc='upper right', title='Sentiment')

    # Save plot to file
    plt.savefig("emotions_filtered_by_thresholds_confidence.png", dpi=300)
    plt.close()


def print_statics_by_emotion():
    """
    Getting basic statistics to try to analyze what threshold to set
    :return:
    """
    df = pd.read_excel("combined_predictions.xlsx")
    emotions = set(df['predicted_emotion'].values)

    for emotion in emotions:
        df_emotion = df[df['predicted_emotion'] == emotion]
        avg_score = df_emotion['positivity_score'].mean()
        std_dev = df_emotion['positivity_score'].std()
        sentiment_counts = df_emotion['label'].value_counts()

        print(f"##########\n"
              f"{emotion}\n"
              f"##########\n")
        print(f"average sentiment score: {avg_score}")
        print(f"std dev of sentiment score: {std_dev}")
        print(sentiment_counts)
        print("")

        thresholds = [(0.1, 0.9), (0.2, 0.8), (0.25, 0.75), (0.3, 0.7)]

        for negativity_threshold, positivity_threshold in thresholds:
            df_emotion_with_threshold = df_emotion[(df_emotion['positivity_score'] > positivity_threshold)
                                                   | (df_emotion['positivity_score'] < negativity_threshold)]
            avg_score = df_emotion_with_threshold['positivity_score'].mean()
            std_dev = df_emotion_with_threshold['positivity_score'].std()
            sentiment_counts = df_emotion_with_threshold['label'].value_counts()

            print(f"positivity thresholds: {positivity_threshold}, negativity threshold: {negativity_threshold}")
            print(f"average sentiment score: {avg_score}")
            print(f"std dev of sentiment score: {std_dev}")
            print(sentiment_counts)
            print("")


NEGATIVE_EMOTIONS = ["fear", "anger", "sadness"]
POSITIVE_EMOTIONS = ["joy"]


def filter_data_based_on_emotion_and_sentiment(positivity_threshold=0.8, negativity_threshold=0.2):
    df = pd.read_excel("combined_predictions.xlsx")
    emotions = set(df['predicted_emotion'].values)

    filtered_negative = df[
        (df['predicted_emotion'].isin(NEGATIVE_EMOTIONS)) &
        # (df['positivity_score'] <= negativity_threshold)
        (df['label'] != "POSITIVE")
        ]

    filtered_positive = df[
        (df['predicted_emotion'].isin(POSITIVE_EMOTIONS)) &
        # (df['positivity_score'] >= positivity_threshold)
        (df['label'] != "NEGATIVE")
        ]

    filtered_df = pd.concat([filtered_negative, filtered_positive], ignore_index=True)

    filtered_df.to_excel("emotions_and_score_threshold_2.xlsx")


if __name__ == '__main__':
    # plot_positivity_by_emotion_with_thresholds()
    # print_statics_by_emotion()
    # filter_data_based_on_emotion_and_sentiment(0.7, 0.3)
    # print_statics_by_emotion()
    plot_positivity_by_emotion("emotions_filtered_by_positivity_label.xlsx", "By Label")
    plot_positivity_by_emotion("emotions_filtered_by_score_threshold.xlsx", "By Threshold")
    plot_positivity_by_emotion_with_thresholds()
