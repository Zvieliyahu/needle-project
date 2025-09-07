import matplotlib.pyplot as plt
from PartyPredictor import *
from analyze_prediction_results_new import evaluate_process_results
##############
## GLOBALS: ##
##############
FILE_PATH = "Data/presidential_speeches.xlsx"
PROCESSED_FILE_PATH = "presidential_speeches_processed.xlsx"
PARTY_PREDICTION_RESULT_FILE_PATH = "predictions_result.xlsx"
BIDEN_FILE_PATH = "Data\cleantext_JoeBiden.tsv"
TRUMP_FILE_PATH = "Data\cleantext_DonaldTrump.tsv"
class PartyAnalyzer:
    """
    Class for analyzing U.S. presidential speeches based on the party.
    """

    def __init__(self, file_path: str = FILE_PATH, classify_speeches: bool = True):
        """
        Initialize a time analyzer instance
        :param file_path: a file path to the original data frame
        :param classify_speeches: boolean to choose if to classify speeches for the first time
        """
        self.file_path_ = file_path
        if classify_speeches:
            process_data(self.file_path_, PROCESSED_FILE_PATH)
            evaluate_process_results(PROCESSED_FILE_PATH)
        self.speeches_df_ = pd.read_excel(PROCESSED_FILE_PATH)


    def plot_speeches_per_party_by_topic(self):
        """
        Plots a graph of percentage of speeches per party (only democratic and republican) by topic.
        :return:
        """
        df_expanded = self.speeches_df_.assign(topic=self.speeches_df_['topics'].str.split(',')).explode('topic')
        df_expanded['topic'] = df_expanded['topic'].str.strip()
        df_expanded = df_expanded[df_expanded['Party'].isin(['Republican', 'Democratic'])]

        # Group by party and topic, count occurrences
        counts = df_expanded.groupby(['Party', 'topic']).size().reset_index(name='count')
        pivot_df = counts.pivot(index='topic', columns='Party', values='count').fillna(0)

        # Calculate total number of speeches per party (across all topics) and calculate percentage
        total_speeches_per_party = pivot_df.sum(axis=0)
        # Calculate the percentage of speeches for each topic by party
        pivot_df_percentage = pivot_df.div(total_speeches_per_party, axis=1) * 100

        # Plot
        ax = pivot_df_percentage.plot(kind='bar', figsize=(16, 8), color=['red', 'blue'])
        plt.title('Percentage of Speeches by Democrats and Republicans for Each Topic', fontsize=16, pad=40)
        plt.suptitle('Mostly similar, biggest differences in Defense, Economy, Labor and Religion', fontsize=12,
                     y=0.90)
        plt.xlabel('Topic', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Party', fontsize=12)
        plt.tight_layout()
        plt.savefig("topic_percentage_by_party.png")

    def plot_speeches_per_party_by_emotion(self):
        """
        Plots a graph of percentage of speeches per party (only democratic and republican) by emotion.
        :return:
        """
        df_filtered = self.speeches_df_[self.speeches_df_['Party'].isin(['Democratic', 'Republican'])]

        # Group by party and emotion
        emotion_party_counts = df_filtered.groupby(['Party', 'predicted_emotion']).size().unstack(fill_value=0)

        # Calculate total number of speeches per party (across all emotions) and calculate percentage
        party_totals = emotion_party_counts.sum(axis=1)

        # Calculate the percentage of speeches for each emotion by party
        emotion_party_percentage = emotion_party_counts.div(party_totals, axis=0) * 100

        # Plot
        ax = emotion_party_percentage.T.plot(kind='bar', figsize=(16, 8), color=['red', 'blue'])
        plt.title('Percentage of Speeches by Democrats and Republicans for Each Emotion', fontsize=16, pad=40)
        plt.suptitle('Similar values for all emotion (both parties)', fontsize=12,
                     y=0.90)
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Party', fontsize=12)
        plt.tight_layout()
        plt.savefig("emotion_percentage_by_party.png")

    def predict_party_test(self):
        """
        Testing prediction on dataset - contains trump and biden (unseen data) speeches.
        :return:
        """
        create_database_vectors(self.speeches_df_)

        # Cleaning Biden data and preparing for test
        df_biden = pd.read_csv(BIDEN_FILE_PATH, sep="\t")
        biden_filtered_df = df_biden[['CleanText', 'Date']]
        biden_filtered_df = biden_filtered_df[biden_filtered_df['CleanText'].notna() & (biden_filtered_df['CleanText'].str.strip() != '')]
        biden_filtered_df['Party'] = 'Democratic'

        # Cleaning Trump data and preparing for test
        df_trump = pd.read_csv(TRUMP_FILE_PATH, sep="\t")
        trump_filtered_df = df_trump[df_trump['SpeechID'].str.startswith('CSPAN', na=False)]
        trump_filtered_df = trump_filtered_df[['CleanText', 'Date']]
        trump_filtered_df = trump_filtered_df[trump_filtered_df['CleanText'].notna() & (trump_filtered_df['CleanText'].str.strip() != '')]
        trump_filtered_df['Party'] = 'Republican'

        # Predicting and vectorising Biden's speeches
        # expanded_features_df_biden = biden_filtered_df['CleanText'].apply(
        #     lambda text: predict_party(text)
        # )
        # expanded_features_df_biden = expanded_features_df_biden.apply(pd.Series)
        # expanded_features_df_biden.columns = FEATURE_COLUMNS
        # biden_filtered_df = pd.concat([biden_filtered_df, expanded_features_df_biden], axis=1)
        biden_filtered_df['predicted_party'] = biden_filtered_df['CleanText'].apply(lambda x: predict_party(x)[0])

        # Only Biden's results (unseen president)
        print("\n######################################################################\n")
        print("Biden's prediction results:")
        # biden_filtered_df['predicted_party'] = biden_filtered_df.apply(test_loss, axis=1)
        misclassification_loss(biden_filtered_df)
        print("\n")

        # Predicting and vectorising Trump's speeches
        # expanded_features_df_trump = trump_filtered_df['CleanText'].apply(
        #     lambda text: predict_party(text)
        # )
        # expanded_features_df_trump = expanded_features_df_trump.apply(pd.Series)
        # expanded_features_df_trump.columns = FEATURE_COLUMNS
        # trump_filtered_df = pd.concat([trump_filtered_df, expanded_features_df_trump], axis=1)

        # Combine datasets
        trump_filtered_df['predicted_party'] = trump_filtered_df['CleanText'].apply(lambda x: predict_party(x)[0])

        filtered_df = pd.concat([biden_filtered_df, trump_filtered_df], ignore_index=True)
        # filtered_df['predicted_party'] = filtered_df.apply(test_loss, axis=1)
        print("Total prediction results:")
        misclassification_loss(filtered_df)
        print("\n######################################################################\n")
        filtered_df.to_excel(PARTY_PREDICTION_RESULT_FILE_PATH)
    def party_analysis(self):
        """
        Running time analysis of presidential speeches over time.
        :return:
        """
        self.plot_speeches_per_party_by_topic()
        self.plot_speeches_per_party_by_emotion()
        self.predict_party_test()