import pandas as pd
import matplotlib.pyplot as plt


def plot_emotions_per_decade(filepath: str, file_title: str):
    """
    Plotting a linear graph of emotions by decade.
    :param filepath: to the Excel file that holds the data
    :param file_title: title of the Excel file
    :return:
    """
    # Load data
    df = pd.read_excel(filepath)

    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid or missing dates
    df = df.dropna(subset=['date'])

    # Create a 'decade' column
    df['decade'] = (df['date'].dt.year // 10) * 10

    # Group by decade and emotion
    emotion_counts = df.groupby(['decade', 'predicted_emotion']).size().reset_index(name='count')

    # Pivot to wide format for plotting
    pivot_df = emotion_counts.pivot(index='decade', columns='predicted_emotion', values='count').fillna(0)

    # Plot
    plt.figure(figsize=(14, 6))
    for emotion in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[emotion], marker='o', label=emotion)

    # Aesthetics
    plt.title('Number of Speeches per Emotion by Decade\n'
              f'{file_title}')
    plt.xlabel('Decade')
    plt.ylabel('Number of Speeches')
    plt.xticks(pivot_df.index, rotation=45)
    plt.legend(title='Emotion')
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"emotions_by_decades_{file_title}")


def plot_avg_positivity_score_per_decade(filepath: str, file_title=""):
    """
    Plotting a linear graph of average positivity score per decade
    :param filepath: a path to an Excel file
    :param file_title: the title of the Excel file
    :return:
    """
    # Load data
    df = pd.read_excel(filepath)

    # Ensure 'date' is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Drop rows with invalid or missing dates
    df = df.dropna(subset=['date'])

    # Create a 'decade' column
    df['decade'] = (df['date'].dt.year // 10) * 10

    # Group by decade and calculate average positivity score
    avg_scores = (
        df.groupby('decade')['positivity_score']
        .mean()
        .reset_index()
    )

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(avg_scores['decade'], avg_scores['positivity_score'], marker='o', color='green', linewidth=2)

    # Aesthetics
    plt.title('Average Positivity Score by Decade\n'
              f'{file_title}')
    plt.xlabel('Decade')
    plt.ylabel('Average Positivity Score')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.xticks(avg_scores['decade'], rotation=45)
    plt.tight_layout()

    # Show the plot
    plt.savefig(f"positivity_score_by_decades_{file_title}")


def plot_topic_counts_by_decade(filepath, directory=""):
    """
    Plotting the number of speeches in a specific topic had been given in each decade.
    :param filepath: a path to the Excel file
    :param directory: the directory path to save to (from file root)
    :return:
    """
    # Load data
    df = pd.read_excel(filepath)

    # Convert date to datetime and drop invalid rows
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'topics'])

    # Extract decade
    df['decade'] = (df['date'].dt.year // 10) * 10

    # Explode topics: split by comma and strip spaces
    df['topics'] = df['topics'].astype(str)
    df['topics_list'] = df['topics'].str.split(',').apply(lambda lst: [t.strip() for t in lst])

    # Explode the dataframe so each topic is its own row
    df_exploded = df.explode('topics_list')

    # Find top 8 topics overall
    top_topics = (
        df_exploded['topics_list']
        .value_counts()
        .nlargest(8)
        .index
        .tolist()
    )

    # Filter only rows with top topics
    df_top = df_exploded[df_exploded['topics_list'].isin(top_topics)]

    # Group by decade and topic, count speeches
    grouped = (
        df_top
        .groupby(['decade', 'topics_list'])
        .size()
        .reset_index(name='speech_count')
        .rename(columns={'topics_list': 'topic'})
    )

    # Pivot for plotting (topics as columns)
    pivot_df = grouped.pivot(index='decade', columns='topic', values='speech_count').fillna(0)

    # Plotting
    plt.figure(figsize=(14, 7))

    for topic in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[topic], marker='o', label=topic)

    plt.title('Number of Speeches Mentioning Each Topic by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Number of Speeches')
    plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(pivot_df.index, rotation=45)
    plt.tight_layout()

    plt.savefig(f"{directory}topics_by_decades.png")


def plot_speeches_per_decade(filepath: str, directory=""):
    """
    Plotting the number of speeches per decade
    :param filepath:
    :param directory:
    :return:
    """
    df = pd.read_excel(filepath)

    # Convert date to datetime and drop invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Extract decade
    df['decade'] = (df['date'].dt.year // 10) * 10

    # Count speeches per decade
    speeches_per_decade = df.groupby('decade').size()

    # Plot
    plt.figure(figsize=(10, 6))
    speeches_per_decade.plot(kind='line', marker='o')

    plt.title('Number of Speeches per Decade')
    plt.xlabel('Decade')
    plt.ylabel('Number of Speeches')
    plt.grid(True)
    plt.xticks(speeches_per_decade.index, rotation=45)
    plt.tight_layout()

    plt.savefig(f"{directory}number_of_speeches_per.png")


def plot_normalized_topics_per_decade(filepath, directory="",top_n=8):
    """
    Plotting linear graph of speeches by topic per decade as a percentage of total speeches of that decade.
    :param filepath:
    :param directory:
    :param top_n: the n'th most common topics (in general)
    :return:
    """
    df = pd.read_excel(filepath)

    df['topics'] = df['topics'].replace(r'^\s*$', pd.NA, regex=True)
    df = df.dropna(subset=['topics'])

    # Convert date to datetime and drop invalid dates
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # Extract decade
    df['decade'] = (df['date'].dt.year // 10) * 10

    # Split topics by comma and explode
    df['topics'] = df['topics'].fillna('')
    df_exp = df.assign(topic=df['topics'].str.split(',')).explode('topic')

    # Clean topic whitespace
    df_exp['topic'] = df_exp['topic'].str.strip()

    # Get top N topics by total count
    top_topics = (
        df_exp['topic']
        .value_counts()
        .nlargest(top_n)
        .index
        .tolist()
    )

    # Filter to top topics only
    df_top = df_exp[df_exp['topic'].isin(top_topics)]

    # Count speeches per topic per decade
    topic_counts = df_top.groupby(['decade', 'topic']).size().unstack(fill_value=0)

    # Count total speeches per decade (including all topics)
    total_speeches = df.groupby('decade').size()

    # Normalize counts by total speeches per decade to get percentages
    topic_percentages = topic_counts.div(total_speeches, axis=0) * 100

    # Plot
    plt.figure(figsize=(12, 7))
    for topic in topic_percentages.columns:
        plt.plot(topic_percentages.index, topic_percentages[topic], marker='o', label=topic)

    plt.title(f'Percentage of Speeches per Topic by Decade (Top {top_n} Topics)')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Speeches (%)')
    plt.legend(title='Topic', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(topic_percentages.index, rotation=45)
    plt.tight_layout()

    plt.savefig(f"{directory}percentage_of_topics_by_decade.png")


"""###Globals:####"""
# FILE_PATH = "emotions_filtered_by_positivity_label.xlsx"
# FILE_NAME = "Data Filtered by Correlation between Emotion and Positivity Label"
# FILE_PATH = "emotions_filtered_by_score_threshold.xlsx"
# FILE_NAME = "Data Filtered by Correlation between Emotion and Positivity Score"
FILE_PATH = "combined_predictions.xlsx"
FILE_DIRECTORY = "Time Analysis/"

if __name__ == '__main__':
    # plot_emotions_per_decade(FILE_PATH, FILE_NAME)
    # plot_avg_positivity_score_per_decade(FILE_PATH, FILE_NAME)
    # plot_topic_counts_by_decade(FILE_PATH, FILE_DIRECTORY)
    # plot_speeches_per_decade(FILE_PATH, FILE_DIRECTORY)
    plot_normalized_topics_per_decade(FILE_PATH,FILE_DIRECTORY)
