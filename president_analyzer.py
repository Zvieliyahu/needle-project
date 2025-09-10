import matplotlib.pyplot as plt
from process_data import *
from analyze_prediction_results import evaluate_process_results
from PCA_president import create_presidents_vectors

##############
## GLOBALS: ##
##############
FILE_PATH = "Data/presidential_speeches.xlsx"
PROCESSED_FILE_PATH = "presidential_speeches_processed.xlsx"


class PresidentAnalyzer:
    """
    Class for analyzing U.S. presidential speeches based on the president.
    """

    def __init__(self, file_path: str = FILE_PATH, classify_speeches: bool = True):
        """
        Initialize a president analyzer instance.
        :param file_path: A file path to the original data frame.
        :param classify_speeches: Boolean to choose if to classify speeches for the first time.
        """
        self.file_path_ = file_path
        if classify_speeches:
            process_data(self.file_path_, PROCESSED_FILE_PATH)
            evaluate_process_results(PROCESSED_FILE_PATH)
        self.speeches_df_ = pd.read_excel(PROCESSED_FILE_PATH)

    def plot_speeches_per_president_by_topic(self):
        """
        Plots a graph of number of speeches per president (only top 10 most active presidents) by topic.
        :return:
        """
        df = self.speeches_df_
        counts = df['President'].value_counts()
        top_10_presidents = counts.head(10).index
        df_top_10 = df[df['President'].isin(top_10_presidents)]
        df_expanded = df_top_10.assign(topic=df_top_10['topics'].str.split(',')).explode('topic')
        df_expanded['topic'] = df_expanded['topic'].str.strip()
        df_expanded['date'] = pd.to_datetime(df_expanded['date'])
        topic_counts = df_expanded.groupby(['President', 'topic']).size().reset_index(name='count')
        presidents = df_top_10['President'].unique()
        for president in presidents:
            # Filter topic data for president
            df_pres_topics = topic_counts[topic_counts['President'] == president].sort_values(by='count',
                                                                                              ascending=False)
            # Get year range for this president
            df_pres_dates = df_top_10[df_top_10['President'] == president]
            min_year = df_pres_dates['date'].dt.year.min()
            max_year = df_pres_dates['date'].dt.year.max()

            # Plot
            plt.figure(figsize=(16, 8))
            plt.bar(df_pres_topics['topic'], df_pres_topics['count'], color='peru')
            plt.title(f"Topics Discussed by {president} ({min_year} - {max_year})",  fontsize=16, pad=40)
            plt.suptitle(f"Visualizing amount of speeches by {president} of each topic", fontsize=12,
                         y=0.90)
            plt.xlabel("Topic")
            plt.ylabel("Number of Speeches")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{president}_topics.png")

    def plot_speeches_per_president_by_emotion(self):
        """
        Plots a graph of number of speeches per president (only top 10 most active presidents) by emotion.
        :return:
        """
        df = self.speeches_df_
        df['date'] = pd.to_datetime(df['date'])
        counts = df['President'].value_counts()
        top_10_presidents = counts.head(10).index
        df_top_10 = df[df['President'].isin(top_10_presidents)]
        emotion_counts = df_top_10.groupby(['President', 'predicted_emotion']).size().reset_index(name='count')
        presidents = df_top_10['President'].unique()
        for president in presidents:
            df_pres_emotions = emotion_counts[emotion_counts['President'] == president].sort_values(by='count',
                                                                                                    ascending=False)

            # Get year range for this president
            df_pres_dates = df_top_10[df_top_10['President'] == president]
            min_year = df_pres_dates['date'].dt.year.min()
            max_year = df_pres_dates['date'].dt.year.max()

            # Plot
            plt.figure(figsize=(16, 8))
            plt.bar(df_pres_emotions['predicted_emotion'], df_pres_emotions['count'], color='violet')
            plt.title(f"Predicted Emotions of Speeches by {president} ({min_year} - {max_year})", fontsize=16, pad=40)
            plt.suptitle(f"Visualizing amount of speeches by {president} of each emotion", fontsize=12,
                         y=0.90)
            plt.xlabel("Predicted Emotion")
            plt.ylabel("Number of Speeches")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{president}_emotions.png")

    def create_database_pca_president(self):
        """
        Creates the database needed for pca of presidents (only top 10 most active presidents,
        only democratic and republican).
        :return:
        """
        df = self.speeches_df_
        df_filtered_parties = df[df['Party'].isin(['Democratic', 'Republican'])]
        president_counts = df_filtered_parties['President'].value_counts()
        top_10_presidents = president_counts.head(10).index
        df = df[df['President'].isin(top_10_presidents)]
        create_presidents_vectors(df)

    def president_analysis(self):
        """
        Running time analysis of presidential speeches over time.
        :return:
        """
        self.plot_speeches_per_president_by_topic()
        self.plot_speeches_per_president_by_emotion()
        self.create_database_pca_president()
