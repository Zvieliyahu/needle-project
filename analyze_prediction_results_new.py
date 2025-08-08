import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib

matplotlib.use('TkAgg')
threshold = 0.6

def print_statics_by_emotion():
    """
    Getting basic statistics to try to analyze what threshold to set for removing emotion predictions.
    :return:
    """
    df = pd.read_excel("emotion_and_positivity_predictions.xlsx")
    emotions = set(df['predicted_emotion'].values)

    for emotion in emotions:
        df_emotion = df[df['predicted_emotion'] == emotion]
        avg_score = df_emotion['confidence'].mean()
        std_dev = df_emotion['confidence'].std()
        sentiment_counts = df_emotion['label'].value_counts()

        print(f"##########\n"
              f"{emotion}\n"
              f"##########\n")
        print(f"average sentiment score: {avg_score}")
        print(f"std dev of sentiment score: {std_dev}")
        print(sentiment_counts)
        print("")

        thresholds = [0.9, 0.8, 0.75, 0.7]

        for threshold in thresholds:
            df_emotion_with_threshold = df_emotion[(df_emotion['confidence'] > threshold)]
            avg_score = df_emotion_with_threshold['confidence'].mean()
            std_dev = df_emotion_with_threshold['confidence'].std()
            sentiment_counts = df_emotion_with_threshold['label'].value_counts()

            print(f"threshold: {threshold}")
            print(f"average sentiment score: {avg_score}")
            print(f"std dev of sentiment score: {std_dev}")
            print(sentiment_counts)
            print("")


NEGATIVE_EMOTIONS = ["fear", "anger", "sadness"]
POSITIVE_EMOTIONS = ["joy"]

def analyze_prediction_results():
    # Filter the data
    # filtered_df = df[df['confidence'] > threshold]
    filtered_df = df

    # Count the predicted_emotions
    emotion_counts = filtered_df['label'].value_counts().sort_values(ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title(f'Predicted Emotions (confidence > {threshold})')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # filtered_df = df[df['confidence'] > threshold]

    # Count the predicted_emotions
    emotion_counts = filtered_df['predicted_emotion'].value_counts().sort_values(ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    emotion_counts.plot(kind='bar', color='skyblue', edgecolor='black')

    plt.title(f'Predicted Emotions (confidence > {threshold})')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # filtered_df = df[df['confidence'] > threshold]

    # Group by predicted_emotion and label
    grouped = filtered_df.groupby(['predicted_emotion', 'label']).size().unstack(fill_value=0)

    # Plot as grouped bar chart
    grouped.plot(kind='bar', figsize=(12, 6), edgecolor='black')

    plt.title(f'Predicted Emotions by Label (confidence > {threshold})')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = pd.read_excel("emotion_and_positivity_predictions.xlsx")
    analyze_prediction_results()