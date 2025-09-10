import pandas as pd
import matplotlib.pyplot as plt
PROCESSED_FILE_PATH = "presidential_speeches_processed.xlsx"


def evaluate_process_results(file_path: str = PROCESSED_FILE_PATH):
    """
    Evaluate the processed data - check whether label and emotion are matching.
    :param file_path: Path to the processed file.
    :return:
    """
    filtered_df = pd.read_excel(file_path)

    # Group by predicted_emotion and label
    grouped = filtered_df.groupby(['predicted_emotion', 'label']).size().unstack(fill_value=0)

    # Color map for the plot
    color_map = {
        'negative': 'red',
        'neutral': 'yellow',
        'positive': 'green'
    }

    # Apply the correct color to each sentiment label
    colors = [color_map.get(emotion, 'gray') for emotion in grouped.columns]

    # Plot
    ax = grouped.plot(kind='bar', figsize=(16, 8), color=colors)
    plt.title('Predicted Emotions by Label', fontsize=16, pad=40)
    plt.suptitle('Mostly matching emotion to label (Joy ~ Positive, Angry ~ Negative, etc\')', fontsize=12, y=0.90)
    plt.xlabel('Predicted Emotion', fontsize=14)
    plt.ylabel('Number of Speeches', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Label', loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.savefig("evaluate_graph_amount_of_emotions_by_label.png")
