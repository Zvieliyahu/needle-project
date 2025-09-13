from filter_topic import *
import os


class TopicClassifier:
    """
    Class for classifying topics of presidential speeches.
    """

    def __init__(self, file_path: str = FILE_PATH, num_appearances: int = MIN_APPEARANCES):
        """
        Initialize a topic classifier instance.
        :param file_path: A file path to the original data set.
        :param num_appearances: A parameter to tune accuracy (the higher, the more accuracy).
        """
        self.file_path_ = file_path
        self.num_appearances_ = num_appearances

    def find_speeches_on_immigration(self):
        """
        Filtering speeches that talks about immigration and output a new dataset with speeches about that topic
        with speeches cutted to the relevant part, with emotions and positivity label assigned to that specific
        part of the speech.
        :return:
        """
        if not os.path.exists(DIRECTORY_IMMIGRATION):
            os.makedirs(DIRECTORY_IMMIGRATION)
        find_speeches_with_keywords(self.file_path_, IMMIGRATION_KEYWORDS, MOST_IMPORTANT_KEYWORDS_IMMIGRATION,
                                    self.num_appearances_, DIRECTORY_IMMIGRATION, "immigration")

    def find_speeches_on_black_rights(self):
        """
        Filtering speeches that talks about black rights and output a new dataset with speeches about that topic
        with speeches cutted to the relevant part, with emotions and positivity label assigned to that specific
        part of the speech.
        :return:
        """
        if not os.path.exists(DIRECTORY_BLACK_RIGHTS):
            os.makedirs(DIRECTORY_BLACK_RIGHTS)
        find_speeches_with_keywords(self.file_path_, BLACK_RIGHTS_KEYWORD, MOST_IMPORTANT_BLACK_RIGHTS_KEYWORDS,
                                    self.num_appearances_, DIRECTORY_BLACK_RIGHTS, "black_rights")

    def find_speeches_on_women_rights(self):
        """
        Filtering speeches that talks about women's rights and output a new dataset with speeches about that topic
        with speeches cutted to the relevant part, with emotions and positivity label assigned to that specific
        part of the speech.
        :return:
        """
        if not os.path.exists(DIRECTORY_WOMEN_RIGHTS):
            os.makedirs(DIRECTORY_WOMEN_RIGHTS)
        find_speeches_with_keywords(self.file_path_, WOMEN_RIGHTS_KEYWORDS, MOST_IMPORTANT_KEYWORDS_WOMEN_RIGHTS,
                                    self.num_appearances_, DIRECTORY_WOMEN_RIGHTS, "womens_rights")

    def find_speeches_on_native_americans(self):
        """
        Filtering speeches that talks about native americans and output a new dataset with speeches about that topic
        with speeches cutted to the relevant part, with emotions and positivity label assigned to that specific
        part of the speech.
        :return:
        """
        if not os.path.exists(DIRECTORY_NATIVE_AMERICANS):
            os.makedirs(DIRECTORY_NATIVE_AMERICANS)
        find_speeches_with_keywords(self.file_path_, NATIVE_AMERICANS_KEYWORDS, MOST_IMPORTANT_KEYWORDS_NATIVE_AMERICANS,
                                    self.num_appearances_, DIRECTORY_NATIVE_AMERICANS, "native_americans")

    def find_speeches_on_wars(self):
        """
        Filtering speeches that talks about wars and output a new dataset with speeches about that topic
        with speeches cutted to the relevant part, with emotions and positivity label assigned to that specific
        part of the speech.
        :return:
        """
        if not os.path.exists(DIRECTORY_WAR):
            os.makedirs(DIRECTORY_WAR)
        find_speeches_with_keywords(self.file_path_, WAR_KEYWORDS, MOST_IMPORTANT_KEYWORDS_WAR,
                                    self.num_appearances_, DIRECTORY_WAR, "war")

    def classify_speeches_by_subject(self):
        """
        Runs topic classification on different subjects.
        :return:
        """
        self.find_speeches_on_immigration()
        self.find_speeches_on_black_rights()
        self.find_speeches_on_women_rights()
        self.find_speeches_on_native_americans()
        self.find_speeches_on_wars()
