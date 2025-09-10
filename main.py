from time_analyzer import *
from party_analyzer import *
from president_analyzer import *

if __name__ == '__main__':
    party_analyzer = PartyAnalyzer(classify_speeches=True)
    party_analyzer.party_analysis()
    time_analyzer = TimeAnalyzer(classify_speeches=True)
    time_analyzer.time_analysis()
    president_analyzer = PresidentAnalyzer(classify_speeches=False)
    president_analyzer.president_analysis()
