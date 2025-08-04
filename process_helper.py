# Positions of tokens to filter for emotion classification:
ALLOWED_POS = {"ADJ", "VERB", "NOUN", "ADV"}


####################################################################################################################

# Common word by topic for classification:
TOPICS_FOR_CLASSIFICATION = {
    "Economy": [
        "economy", "economic", "jobs", "job", "employment", "inflation", "deficit", "deficits", "recession",
        "growth", "market", "trade", "trades", "trading", "finance", "tax", "taxes", "stimulus", "income",
        "unemployment", "commerce"
    ],
    "Healthcare": [
        "healthcare", "insurance", "patients", "patient", "hospitals", "hospital", "medicare", "medicaid", "affordable",
        "doctors", "doctor", "nurses", "nurse", "medical", "coverage", "public health", "epidemic", "pandemic",
        "treatments", "treatment"
    ],
    "Education": [
        "education", "school", "schools", "students", "student", "teachers", "teacher", "learn", "learns",
        "learning", "curriculum", "college", "colleges", "universities", "scholarships"
                                                                         "university", "scholarship", "literacy",
        "classroom"
    ],
    "Defense": [
        "military", "defense", "forces", "force", "troops", "troop", "security", "war", "weapon", "weapons", "army",
        "navy", "air force", "nuclear", "terrorism", "strategy", "homeland", "counterterrorism", "terrorist"
    ],
    "Foreign Policy": [
        "foreign", "diplomacy", "treaty", "allies", "ambassador", "international", "relations",
        "sanctions", "war", "conflict", "aid", "peacekeeping", "global", "coalition"
    ],
    "Immigration": [
        "immigration", "migrant", "migrants", "border", "borders", "asylum", "visa", "refugee", "refugees",
        "citizenship", "deportation", "illegal", "undocumented", "green card", "immigrants", "immigrant", "amnesty"
    ],
    "Civil Rights": [
        "civil", "rights", "equality", "justice", "discrimination", "racism", "segregation", "freedom",
        "injustice", "prejudice", "black", "minorities", "voting rights", "LGBTQ", "gender", "genders"
    ],
    "Technology": [
        "technology", "innovation", "researches", "research", "science", "AI", "artificial",
        "cybersecurity", "internet", "data", "automation", "engineering", "robotics"
    ],
    "Law and Order": [
        "crime", "crimes", "justice", "police", "enforcement", "court", "courts", "prisons", "prison",
        "incarceration", "drug", "guns", "lawsuits", "trials"
                                                     "safety", "homicide", "drugs", "violence", "gun", "lawsuit",
        "trial", "judge"
    ],
    "Social Security & Welfare": [
        "social", "welfare", "benefit", "benefits", "retirement", "pensions", "safety",
        "assistance", "disability", "unemployment", "insurance", "stamps",
        "slaves", "slave", "slavery"
    ],
    "Gun Control": [
        "gun", "firearm", "firearms", "shooting", "amendment", "NRA", "checks",
        "violence", "weapon", "assault", "control", "shooter"
    ],
    "Women’s Rights": [
        "women", "gender", "reproductive", "abortion", "equality", "women's", "maternity",
        "equal", "rights"
    ],
    "Labor and Unions": [
        "labor", "union", "worker", "workers", "wages", "strike", "collective", "bargaining", "overtime",
        "minimum", "rights", "collar", "wage"
    ],
    "Religion": [
        "faith", "religion", "church", "god", "christian", "muslim", "jewish", "freedom",
        "worship", "religious", "spiritual", "belief"
    ],
    "Racism": ["racism", "racial", "discrimination", "prejudice", "segregation", "inequality",
               "bias", "bigotry", "justice", "equality",
               "profiling", "civil rights", "racial violence",
               "oppression", "supremacy", "racial disparity",
               "social", "brutality", "segregation"
               ],
    "Environment": [
        "environment", "climate", "pollution", "carbon", "emissions", "warming", "green",
        "sustainability", "renewable", "energy", "solar", "wind", "clean air", "clean water"
    ],
    "Agriculture": [
        "farming", "agriculture", "rural", "farmers", "crops", "livestock", "subsidy", "drought",
        "harvest", "agricultural", "soil", "irrigation"
    ],
    "Energy": [
        "energy", "oil", "gas", "renewable", "electricity", "fuels", "solar",
        "wind", "efficiency", "fuel"
    ],
    "Infrastructure": [
        "infrastructure", "road", "roads", "bridge", "bridges", "highway", "highways", "transportation", "transit",
        "rail",
        "airport", "airports", "construction", "public works", "telecom", "broadband", "utilities"
    ]
}
labels = [
    "Economy", "Foreign Policy", "National Security", "Civil Rights", "Healthcare",
    "Education", "Environment", "Law and Order", "Government and Institutions",
    "Technology and Innovation", "Religion and Morality", "Campaign and Politics"
]


####################################################################################################################


# Unrelated words to ignore when classifying topics
UNRELATED_TOPIC_WORDS = [
    "go", "think", "want", "know", "say", "good", "lot", "get", "like",
    "thank", "time", "people", "work", "today", "thing", "great", "come",
    "look", "way", "right", "help", "need", "make", "well", "let", "tell",
    "see", "take", "try", "keep", "talk", "ask", "use", "give", "put",
    "feel", "seem", "leave", "mean", "start", "call", "show", "really",
    "big", "year", "new", "last", "many", "still", "find", "even",

    # Additional filler/generic/common words often not topic-specific:
    "also", "much", "much", "very", "always", "never", "quite", "just",
    "lot", "some", "most", "more", "most", "better", "better", "much",
    "well", "today", "tonight", "okay", "ok", "yes", "no", "let's", "us",
    "got", "getting", "did", "didn't", "does", "doesn't", "do", "don't",
    "will", "would", "could", "should", "might", "must", "can", "shall",
    "probably", "maybe", "actually", "basically", "seriously", "honestly",
    "definitely", "literally", "sure", "right", "okay", "alright", "hey",
    "hello", "hi", "well", "oh", "um", "uh", "yeah", "okay", "anyway",
    "anyways", "however", "therefore", "thus", "meanwhile", "although",
    "though", "yet", "still", "actually", "simply", "quite", "pretty",
    "rather", "soon", "often", "always", "never", "sometimes", "usually",
    "again", "already", "soon", "later", "ago",

    # Common pronouns (sometimes too generic for topic detection):
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",

    # Common conjunctions and prepositions that spaCy usually removes but can add:
    "and", "or", "but", "if", "because", "while", "as", "until", "when",
    "where", "after", "before", "since", "though", "although", "nor",
    "so", "than", "whether",

    # Other words we've missed
    "president", "go", "united", "states", "america", "country", "say", "nation", "world", "american"
]
