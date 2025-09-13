from topic_classifier import *
from war_speeches_analysis import *
from time_analyzer_plot_helper import *
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TimeAnalyzer:
    """
    Class for analyzing U.S. presidential speeches over time.
    """

    def __init__(self, file_path: str = FILE_PATH, classify_speeches: bool = True):
        """
        Initialize a time analyzer instance.
        :param file_path: A file path to the original data frame.
        :param classify_speeches: Boolean to choose if to classify speeches for the first time.
        """
        self.speeches_df_ = pd.read_excel(file_path)
        if classify_speeches:
            topic_classifier_ = TopicClassifier(file_path, MIN_APPEARANCES)
            topic_classifier_.classify_speeches_by_subject()

    @staticmethod
    def plot_speeches_per_decade_by_topic():
        """
        Plots a graph of amount of speeches by selected topic per decade.
        :return:
        """
        df_immigration = pd.read_excel(PATH_IMMIGRATION)
        df_black_rights = pd.read_excel(PATH_BLACK_RIGHTS)
        df_native_americans = pd.read_excel(PATH_NATIVE_AMERICANS)
        df_women_rights = pd.read_excel(PATH_WOMEN_RIGHTS)

        topic_dfs = {
            "Immigration": df_immigration,
            "Black Rights": df_black_rights,
            "Native Americans": df_native_americans,
            "Women's Rights": df_women_rights
        }

        plot_speeches_per_decade_by_topic(topic_dfs)

    @staticmethod
    def plot_war_and_peace_terminology_use():
        """
        Plotting a graph of use of peace related words and war related wars over the years in presidential
        speeches on war.
        :return:
        """
        df_war = pd.read_excel(PATH_WAR)
        df_war = df_war[df_war['speech'].notna()]
        start_year = 1850
        end_year = 2020
        plot_keyword_trends_per_year(df_war, start_year=start_year, end_year=end_year, war_keywords=WAR_MORAL_WORDS,
                                     peace_keywords=WAR_RESOLVE_WORDS, war_periods=WAR_PERIODS, rolling_window=5,
                                     output_path="war/usage_of_peace_and_war.png")

    @staticmethod
    def compare_speeches_similarity(df, topic, title_1, years_1, title_2, years_2):
        """
        Comparing difference in rhetoric between two periods.
        :return:
        """
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year

        def assign_period(year):
            """
            Assigning periods of each immigration wave.
            :param year:
            :return:
            """
            if years_1[0] <= year <= years_1[1]:
                return title_1
            elif years_2[0] <= year <= years_2[1]:
                return title_2
            else:
                return "other"

        df['period'] = df['year'].apply(assign_period)

        # Filter only relevant speeches
        df = df[df['period'] != "other"]
        contingency = pd.crosstab(df['period'], df['confidence'])
        chi2, p, dof, ex = chi2_contingency(contingency)

        print("\n######################################################################\n")
        print(f"{topic}:")

        print(contingency)
        print("Chi2:", chi2, "p-value:", p)

        contingency = pd.crosstab(df['period'], df['predicted_emotion'])
        chi2, p, dof, ex = chi2_contingency(contingency)
        contingency.to_excel(f"{topic}_{title_1}_{title_2}_contingency.xlsx")
        print(contingency)
        print("Chi2:", chi2, "p-value:", p)

        emotion_similarity = pd.crosstab(df['period'], df['predicted_emotion'], normalize='index')
        sim = cosine_similarity([emotion_similarity.loc[title_1]], [emotion_similarity.loc[title_2]]).item()

        print("Cosine similarity in emotion space:", sim)

        # Create TF-IDF matrix
        df = df.reset_index(drop=True)

        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(df['speech'])

        # Average vectors per period
        period_means = df.groupby("period").apply(
            lambda x: np.mean(X[x.index].toarray(), axis=0),
            include_groups=False
        )

        # Cosine similarity between the two periods
        cos_sim = cosine_similarity([period_means.iloc[0]], [period_means.iloc[1]])
        print("Cosine similarity (text-based):", cos_sim[0][0])

        print("\n######################################################################\n")

    def compare_wars(self):
        """
        Runs topic analysis for two periods of wars: World War 2 and Vietnam War.
        :return:
        """
        df_war = pd.read_excel(PATH_WAR)
        df_war = df_war[df_war['speech'].notna()]

        word_clouds = []
        # Plot cloud tag
        for war in [("ww2", 1939, 1945), ("vietnam", 1965, 1973)]:
            output_path = f"war/word_cloud_tf-idf_{war[0]}.png"
            plot_tfidf_word_cloud(df_war, self.speeches_df_, start_year=war[1], end_year=war[2],
                                  output_path=output_path, topic=war[0])
            word_clouds.append(output_path)
        combine_images_side_by_side(word_clouds, output_path="war/word_comparison.png")

        label, pre_output_path, normalize_value, y_axis_limit, keywords = None, None, None, None, None
        for topic in TOPICS:
            match topic:
                case "VICTORY":
                    label = VICTORY_PRE_LABEL
                    pre_output_path = VICTORY_PRE_PATH
                    normalize_value = NORMALIZE_VALUE_VICTORY
                    y_axis_limit = Y_AXIS_LIMIT_VICTORY
                    keywords = VICTORY_WORDS
                case "PEACE":
                    label = PEACE_PRE_LABEL
                    pre_output_path = PEACE_PRE_PATH
                    normalize_value = NORMALIZE_VALUE_PEACE
                    y_axis_limit = Y_AXIS_LIMIT_PEACE
                    keywords = PEACE_WORDS
                case "ECONOMY":
                    label = ECONOMY_PRE_LABEL
                    pre_output_path = ECONOMY_PRE_PATH
                    normalize_value = NORMALIZE_VALUE_ECONOMY
                    y_axis_limit = Y_AXIS_LIMIT_ECONOMY
                    keywords = ECONOMY_WORDS

            plots = []
            # Plot the plots for each of the wars
            for war, file_title, periods in LIST_OF_PERIODS:
                plot_title = f"{label} {war}"
                output_path = f"{pre_output_path}{file_title}.png"
                plots.append(output_path)

                plot_use_of_keyword_over_periods(
                    df=df_war,
                    periods=periods,
                    normalize_value=normalize_value,
                    y_axis_limit=y_axis_limit,
                    plot_title=plot_title,
                    keywords=keywords,
                    output_path=output_path
                )

            # Put both plots side by side
            combine_images_side_by_side(plots, output_path=f"war/{topic}_merged.png")

    def war_analysis(self):
        """
        Creating war analysis plots.
        :return:
        """
        self.plot_war_and_peace_terminology_use()
        self.compare_wars()

    def plot_black_rights_word_cloud(self):
        """
        Plotting word cloud by periods of the struggle for black rights.
        :return:
        """
        df_black_rights = pd.read_excel(PATH_BLACK_RIGHTS)
        plots = []
        for time_period in [("Emancipation", 1850, 1870), ("Segregation", 1870, 1950),
                            ("Civil Rights Movement", 1950, 1970)]:
            output_path = f"black rights/word_cloud_tf-idf_{time_period[0]}.png"
            plot_tfidf_word_cloud(df_black_rights, self.speeches_df_, start_year=time_period[1],
                                  end_year=time_period[2], topic=time_period[0],
                                  output_path=output_path)
            plots.append(output_path)

        combine_images_side_by_side(plots, output_path="black rights/word_clouds.png")

    @staticmethod
    def get_emotion_stats_black_rights():
        """
        Printing emotions and positivity labels statistics.
        :return:
        """
        df_black_rights = pd.read_excel(PATH_BLACK_RIGHTS)
        print("\n######################################################################\n")
        print("Black Rights:")
        emotion_counts = df_black_rights['predicted_emotion'].value_counts()
        print("Predicted emotions counts:")
        print(emotion_counts)

        label_counts = df_black_rights['label'].value_counts()
        print("\nLabel counts:")
        print(label_counts)

        emotion_label_table = pd.crosstab(df_black_rights['label'], df_black_rights['predicted_emotion'])
        print("\nEmotion distribution per label:")
        print(emotion_label_table)

        chi2, p, dof, expected = chi2_contingency(emotion_label_table)
        print("\nChi square test for independence between labels and emotions")
        print("Chi^2:", chi2, "p-value:", p)
        print("\n######################################################################\n")

    def plot_immigration_word_cloud(self):
        """
        Plotting word cloud by periods of the struggle for black rights.
        :return:
        """
        df_immigration = pd.read_excel(PATH_IMMIGRATION)
        plots = []
        for time_period in [("Chinese Immigration Wave", 1870, 1890), ("Current Immigration Wave", 2000, 2020)]:
            output_path = f"immigration/word_cloud_tf-idf_{time_period[0]}.png"
            plot_tfidf_word_cloud(df_immigration, self.speeches_df_, start_year=time_period[1],
                                  end_year=time_period[2], topic=time_period[0],
                                  output_path=output_path)
            plots.append(output_path)

        combine_images_side_by_side(plots, output_path="immigration/word_clouds.png")

    def similarity_by_decade(self):
        """
        Plots a cosine similarity per decade with respect to modern speeches (2000 - 2020).
        :return:
        """
        # Adding decade column
        df = self.speeches_df_
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['decade'] = (df['year'] // 10) * 10

        # Compute TF-IDF
        vectorizer = TfidfVectorizer(stop_words="english")
        X = vectorizer.fit_transform(df['speech'])

        # Reset index for alignment with X
        df = df.reset_index(drop=True)

        # Average vector for reference period
        ref_idx = df[(df['year'] >= 2000) & (df['year'] <= 2020)].index
        ref_vector = np.mean(X[ref_idx].toarray(), axis=0).reshape(1, -1)

        # Compute similarity for each decade
        similarities = {}
        for decade in sorted(df['decade'].unique()):
            if decade == 2020:  # Empty decade
                continue
            decade_idx = df[df['decade'] == decade].index
            if len(decade_idx) > 0:
                decade_vector = np.mean(X[decade_idx].toarray(), axis=0).reshape(1, -1)
                sim = cosine_similarity(ref_vector, decade_vector)[0][0]
                similarities[decade] = sim

        # Converting to DataFrame
        sim_df = pd.DataFrame(list(similarities.items()), columns=['decade', 'similarity']).sort_values('decade')

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Textual similarity of presidential speeches to 2000–2020 reference")

        ax.text(
            0.01, 1.02,
            LANGUAGE_CHANGE_CAPTION,
            transform=ax.transAxes,
            fontsize=10, color='gray', ha='left', va='bottom'
        )

        plt.plot(sim_df['decade'], sim_df['similarity'])
        plt.axhline(1.0, color='gray', linestyle='--', label="Perfect similarity")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Cosine similarity")
        plt.xticks(sim_df['decade'], rotation=45)
        plt.grid(True)
        plt.legend()
        plt.savefig("similarity_by_decade.png")

    def print_stats(self):
        """
        Print summary of statistics used in time analysis.
        :return:
        """
        # Header
        topic = "* Topic Statistical Analysis *"
        print("*" * (len("Topic Statistical Analysis") + 4))
        print(topic)
        print("*" * (len("Topic Statistical Analysis") + 4))

        self.compare_speeches_similarity(pd.read_excel(PATH_IMMIGRATION), "Immigration", "Chinese Immigration wave",
                                         (1870, 1890),
                                         "Modern Immigration Wave", (2000, 2020))

        self.compare_speeches_similarity(pd.read_excel(PATH_WAR), "War", "World War 2", (1939, 1945),
                                         "Vietnam War", (1963, 1975))
        self.get_emotion_stats_black_rights()

    def time_analysis(self):
        """
        Running time analysis of presidential speeches over time.
        :return:
        """
        self.plot_speeches_per_decade_by_topic()
        self.war_analysis()
        self.plot_black_rights_word_cloud()
        self.plot_immigration_word_cloud()
        self.similarity_by_decade()
        self.print_stats()
