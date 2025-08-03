import pandas as pd
from textblob import TextBlob
from nrclex import NRCLex

topics = {
    "Economy": [
        "economy", "economic", "jobs", "job", "employment", "inflation", "deficit", "deficits", "recession",
        "growth", "market", "trade", "trades", "trading", "finance", "tax", "taxes", "stimulus", "income",
        "unemployment"
    ],
    "Healthcare": [
        "healthcare", "insurance", "patients", "patient", "hospitals", "hospital", "medicare", "medicaid", "affordable",
        "doctors", "doctor", "nurses", "nurse", "medical", "coverage", "public health", "epidemic", "pandemic",
        "treatments","treatment"
    ],
    "Education": [
        "education", "school", "schools", "students", "student", "teachers", "teacher", "learn", "learns",
        "learning", "curriculum", "college", "colleges", "universities", "scholarships" 
        "university", "scholarship", "literacy", "classroom"
    ],
    "Environment": [
        "environment", "climate", "pollution", "carbon", "emissions", "warming", "green",
        "sustainability", "renewable", "energy", "solar", "wind", "clean air", "clean water"
    ],
    "Defense": [
        "military", "defense", "forces", "force", "troops", "troop", "security", "war", "weapon", "weapons", "army",
        "navy", "air force", "nuclear", "terrorism", "strategy", "homeland", "counterterrorism"
    ],
    "Foreign Policy": [
        "foreign", "diplomacy", "treaty", "allies", "ambassador", "international", "relations",
        "sanctions", "war", "conflict", "aid", "peacekeeping", "global", "coalition"
    ],
    "Immigration": [
        "immigration", "migrant", "migrants", "border", "borders", "asylum", "visa", "refugee", "refugees",
        "citizenship", "deportation", "illegal", "undocumented", "green card", "immigrants","immigrant", "amnesty"
    ],
    "Civil Rights": [
        "civil", "rights", "equality", "justice", "discrimination", "racism", "segregation", "freedom",
        "injustice", "prejudice", "black", "minorities", "voting rights", "LGBTQ", "gender", "genders"
    ],
    "Technology": [
        "technology", "innovation", "researches","research", "science", "AI", "artificial",
        "cybersecurity", "internet", "data", "automation", "engineering", "robotics"
    ],
    "Energy": [
        "energy", "oil", "gas", "renewable", "power", "electricity", "fuels", "solar",
        "wind", "clean energy", "nuclear energy", "efficiency", "fuel"
    ],
    "Infrastructure": [
        "infrastructure", "road","roads", "bridge","bridges", "highway" ,"highways", "transportation", "transit", "rail",
        "airport","airports", "construction", "public works", "telecom", "broadband", "utilities"
    ],
    "Law and Order": [
        "crime", "crimes", "justice", "police", "enforcement", "court", "courts", "prisons", "prison",
        "incarceration", "drug", "guns", "lawsuits", "trials"
        "safety", "homicide", "drugs", "violence", "gun", "lawsuit", "trial", "judge"
    ],
    "Social Security & Welfare": [
        "social", "welfare", "benefit", "benefits", "retirement", "pensions", "safety",
        "assistance", "disability", "unemployment", "insurance", "stamps"
    ],
    "Veterans": [
        "veterans", "veteran", "VA", "benefits", "military service", "healthcare", "support", "disabled",
        "deployment", "honor", "sacrifice", "transition", "jobs for veterans"
    ],
    "Gun Control": [
        "gun", "firearm","firearms", "shooting", "amendment", "NRA", "checks",
        "violence", "weapon", "assault", "control", "shooter"
    ],
    "Women’s Rights": [
        "women", "gender", "reproductive", "abortion", "equality", "women's", "maternity",
        "equal", "rights"
    ],
    "Labor and Unions": [
        "labor", "union", "worker", "workers", "wages", "strike", "collective","bargaining", "overtime",
        "minimum", "rights", "collar", "wage"
    ],
    "Religion": [
        "faith", "religion", "church", "god", "christian", "muslim", "jewish", "freedom",
        "worship", "religious", "spiritual", "belief"
    ],
    "Agriculture": [
        "farming", "agriculture", "rural", "farmers", "crops", "livestock", "subsidy", "drought",
        "harvest", "agricultural", "soil", "irrigation"
    ],
    "Transportation": [
        "transportation", "traffic", "transit", "infrastructure", "rail", "bus", "highways",
        "aviation", "freight", "mobility", "logistics"
    ],
    "Racism": ["racism", "racial", "discrimination", "prejudice", "segregation", "inequality",
    "bias", "bigotry", "justice", "equality",
    "profiling", "civil rights", "racial violence",
    "oppression", "supremacy", "racial disparity",
    "social", "brutality", "segregation"
    ]
}
labels = [
    "Economy", "Foreign Policy", "National Security", "Civil Rights", "Healthcare",
    "Education", "Environment", "Law and Order", "Government and Institutions",
    "Technology and Innovation", "Religion and Morality", "Campaign and Politics"
]
# Extract sentiment per speech
def extract_sentiments(speech : str) -> float:
    blob = TextBlob(speech)
    sentiment = blob.sentiment.polarity
    # possibly process more
    return sentiment

def extract_emotions (speech : str) -> dict:
    emotion = NRCLex(speech)
    print(emotion.raw_emotion_scores)

def extract_topic(speech:str)->list:
    tokens = speech.split()  # tokenize by spaces
    matched_topics = {}
    for topic, keywords in topics.items():
        found_keywords = [kw for kw in keywords if kw in tokens]
        if found_keywords:
            matched_topics[topic] = {
                "count": len(found_keywords),
                "keywords": found_keywords
            }

    # Sort by number of matched keywords descending
    matched_topics = dict(sorted(matched_topics.items(), key=lambda x: x[1]["count"], reverse=True))
    filtered_topics = {topic: data for topic, data in matched_topics.items() if data["count"] > 2}
    filtered_topic_names = list(filtered_topics.keys())
    return filtered_topic_names