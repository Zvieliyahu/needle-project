from transformers import pipeline

##############
## GLOBALS: ##
##############
CHUNK_SIZE = 200
LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}
MIN_APPEARANCES = 8
FILE_PATH = 'combined_predictions.xlsx'
DIRECTORY = 'immigration/'

##################
## CLASSIFIERS: ##
##################
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1,
    truncation=True
)

sentiment_pipeline = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    top_k=2,
    truncation=True
)

"""
                                   #################
                                   ## IMMIGRATION ##
                                   #################
"""
MOST_IMPORTANT_KEYWORDS_IMMIGRATION = ["immigration", "immigrant", "immigrants", "migrant", "migrants"]

IMMIGRATION_KEYWORDS = [
    "immigration", "immigrant", "immigrants", "migrant", "migrants",
    "asylum", "refugee", "refugees", "border", "borders", "deportation",
    "visa", "visas", "green card", "citizenship", "naturalization",
    "illegal", "undocumented", "alien", "foreigners",
    "entry", "emigration", "migration", "detention", "amnesty",
    "border security", "wall", "open borders", "closed borders",
    "customs", "ICE", "CBP", "dreamers", "DACA", "sanctuary", "anchor baby",
    "nationality", "residency", "permanent resident", "expatriate", "expat",
    "family reunification", "immigration reform", "chain migration",
    "work permit", "temporary protected status", "TPS"
]
