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
FILE_PATH = 'Data/presidential_speeches.xlsx'

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
FILE_PATH_IMMIGRATION = 'combined_predictions.xlsx'
DIRECTORY_IMMIGRATION = 'immigration/'

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


"""
                                   ##################
                                   ## Black Rights ##
                                   ##################
"""
FILE_PATH_BLACK_RIGHTS = 'Data/presidential_speeches.xlsx'
DIRECTORY_BLACK_RIGHTS = 'black rights/'

MOST_IMPORTANT_BLACK_RIGHTS_KEYWORDS = [
    "black rights", "black americans", "african americans", "black community",
    "freedmen", "the negro race", "negro rights", "colored people", "negro", "negroes",
    "slaves", "slavery", "slave"
]

BLACK_RIGHTS_KEYWORD = [
    # General modern terms (Post-1950s civil rights)
    "civil rights", "racial equality", "black rights", "segregation", "desegregation",
    "integration", "affirmative action", "discrimination", "racial justice",
    "racial prejudice", "systemic racism", "equal protection", "racial inequality",
    "equal opportunity", "black community", "black americans", "african americans",
    "minorities", "people of color", "racial profiling",

    # Civil Rights Movement & related figures (1950s–70s)
    "martin luther king", "mlk", "i have a dream", "march on washington",
    "rosa parks", "selma", "montgomery", "civil rights act", "voting rights act",
    "jim crow", "freedom riders", "naacp", "black power", "black panthers",
    "malcolm x",

    # Reconstruction & post-Civil War (1865–1900)
    "freedmen", "emancipation", "freedmens bureau", "abolition", "13th amendment",
    "14th amendment", "15th amendment", "reconstruction", "black codes",
    "ku klux klan", "kkk", "lynching", "segregated", "plessy v. ferguson",

    # Slavery era (1790–1865)
    "slavery", "slave", "slaves", "enslaved", "plantation", "abolitionist",
    "emancipation proclamation", "fugitive slave", "underground railroad",
    "dred scott", "uncle tom", "slave trade", "manumission",

    # Older/historical offensive or coded terms (USE WITH CAUTION)
    "negro", "colored", "colored people", "the negro race", "mulatto", "man of color",
    "servants", "negroes"
]


"""
                                   ####################
                                   ## Women's Rights ##
                                   ####################
"""
DIRECTORY_WOMEN_RIGHTS = 'women rights/'

WOMEN_RIGHTS_KEYWORDS = [
    # Basic women identifiers (all forms, plural and singular)
    #"woman", "women", "female", "females", "lady", "ladies", "girl", "girls", "mother", "mothers", "daughter", "daughters", "wife", "wives",

    # Voting and political participation (all related terms)
    # "suffrage", "suffragette", "right to vote", "voting rights", "ballot", "ballot box", "vote", "votes", "nineteenth amendment", "equal franchise",

    # Equality and justice terms (general and gender-related)
    "equal rights", "equality", "gender equality", "civil rights", "social justice", "human rights", "discrimination", "sex discrimination", "gender discrimination", "equal opportunity", "fair treatment",

    # Workplace, labor, and economic terms
    "workplace", "working women", "working mothers", "employment", "job opportunity", "equal pay", "pay equity", "wage gap", "labor rights", "economic empowerment", "female labor", "female workforce",

    # Education and training
    "education", "education for women", "female education", "girls education", "schools for girls", "higher education", "access to education", "training",

    # Family, motherhood, and caregiving
    "motherhood", "maternal health", "maternity leave", "child care", "childcare", "child rearing", "domestic responsibilities", "home life", "housewife", "homemaker", "caregiver",

    # Legal and political reforms
    "women’s rights", "womens rights", "women’s liberation", "womens liberation", "equal rights amendment", "era", "title ix", "women’s caucus", "gender justice",

    # Social movements and activism
    "feminism", "feminist", "women’s movement", "womens movement", "activism", "advocacy", "women’s organizations", "womens organizations", "temperance movement", "abolition movement",

    # Violence and protection issues
    "domestic violence", "violence against women", "sexual harassment", "sexual assault", "rape crisis", "protection of women",

    # Health and reproductive rights
    "reproductive rights", "birth control", "family planning", "abortion rights", "women’s health", "maternal mortality",

    # Military and civic participation
    "women in the military", "military women", "servicewomen", "women veterans", "civic participation",

    # Cultural and societal roles
    "social roles", "gender roles", "femininity", "womanhood", "female leadership", "role of women",

    # Modern gender equality terms
    "glass ceiling", "sexual equality", "gender pay gap", "womens empowerment", "equal representation", "gender balance", "gender parity", "gender justice"
]

MOST_IMPORTANT_KEYWORDS_WOMEN_RIGHTS = [
    "woman", "women", "girl", "girls", "mother", "mothers", "voting rights"
    "womens rights", "womens suffrage", "nineteenth amendment",
    "equal pay", "equal pay for women", "gender equality", "reproductive rights",
    "female empowerment", "violence against women", "sexual harassment",
    "women in the workforce", "women in the military", "equal rights amendment",
    "title ix", "sex based discrimination", "feminist movement", "womens liberation",
    "working womens rights", "maternal health", "maternity leave", "equal opportunity for women",
    "glass ceiling", "womens economic empowerment", "gender pay gap", "womens political participation",
    "womens education rights", "womens voting rights", "domestic violence", "sexual assault", "family planning"
]


"""
                                   ####################
                                   ## Women's Rights ##
                                   ####################
"""
DIRECTORY_NATIVE_AMERICANS = 'native americans/'

NATIVE_AMERICANS_KEYWORDS = [
    # People identifiers
    "native american", "native americans", "indian", "indians", "american indian", "american indians",
    "first nations", "aboriginal", "tribe", "tribes", "tribal", "reservation", "reservations",
    "indigenous people", "indigenous peoples", "indigenous",

    # Tribal nations and groups (common examples)
    "cherokee", "sioux", "apache", "navajo", "chippewa", "iroquois", "creek", "blackfeet",
    "cheyenne", "shawnee", "seminole", "hopi", "nez perce", "pueblo", "lakota",

    # Key issues and concepts
    "tribal sovereignty", "land rights", "land claim", "land claims",
    "federal recognition", "indian removal", "indian affairs", "indian policy",
    "indian bureau", "indian agent", "indian school",


    # Legal and political terms
    "indian law", "indian act", "tribal law", "indian civil rights act", "indian gaming",
    "indian health service", "indian education",

    # Historical events and policies
    "trail of tears", "dawes act", "indian removal act", "termination policy",
    "native american rights movement", "american indian movement",

    # Cultural and spiritual terms
    "powwow", "native culture", "native traditions", "language preservation",
    "spirituality", "traditional knowledge",

    # Contemporary issues
    "land reclamation", "pipeline protests", "dakota access pipeline", "environmental justice"

    # Extra words
    "savages", "savage"
]

MOST_IMPORTANT_KEYWORDS_NATIVE_AMERICANS = [
    "indian", "indians"
    "tribal sovereignty", "land rights", "land claims", "treaty rights", "self determination",
    "federal recognition", "indian removal act", "dawes act", "indian civil rights act",
    "native american rights movement", "american indian movement", "indian health service",
    "indian education", "tribal law", "indian gaming", "tribal government", "tribal council",
    "indigenous rights", "native american treaty", "tribal treaty",
    "boarding school", "trail of tears", "termination policy", "native american poverty",
    "native american unemployment", "native american education", "native american healthcare",
    "pipeline protests", "dakota access pipeline", "environmental justice"
]

