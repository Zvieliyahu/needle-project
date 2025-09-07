import pandas as pd
from cleanData import *
# from PartyPredictor import *
from process_data import *
from time_analyzer import *
from party_analyzer import *
from president_analyzer import *
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # cleaned_presidential_speeches = clean_presidential_speeches('Data\presidential_speeches.xlsx')
    # create_database_vectors(cleaned_presidential_speeches)
    # speeches_df = pd.read_excel('emotion_and_positivity_predictions.xlsx')
    # speeches_df['topics'] = speeches_df['speech'].apply(classify_topic)
    # speeches_df.to_excel("topics_emotion_and_positivity_predictions.xlsx")
    party_analyzer = PartyAnalyzer(classify_speeches=False)
    party_analyzer.party_analysis()
    time_analyzer = TimeAnalyzer(classify_speeches=False)
    time_analyzer.time_analysis()
    president_analyzer = PresidentAnalyzer(classify_speeches=False)
    president_analyzer.president_analysis()