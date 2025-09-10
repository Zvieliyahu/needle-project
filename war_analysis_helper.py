"""
                  ###############
                  ##  GLOBALS  ##
                  ###############
"""
# Plot
Y_LABEL = "Words Count"
X_LABEL = "Period in War"

# Time Periods
VIETNAM_PERIODS = [{"Pre-war": (1960, 1963)},
                   {"War Time\nPresident Lyndon B. Johnson": (1964, 1968)},
                   {"War Time\nPresident Richard M. Nixon": (1969, 1974)}, {"Post-war": (1975, 1980)}]
WW2_PERIODS = [{"Pre-war": (1935, 1938)}, {"War time": (1939, 1945)}, {"Post-war": (1946, 1950)}]
LIST_OF_PERIODS = [("Vietnam War", "periods_of_vietnam_war", VIETNAM_PERIODS),
                   ("World War 2", "periods_of_ww2", WW2_PERIODS)]

# TOPICS:
VICTORY_PRE_LABEL = "Victory Word Usage in"
VICTORY_PRE_PATH = "war/victory_by_"
NORMALIZE_VALUE_VICTORY = 10
Y_AXIS_LIMIT_VICTORY = 350

PEACE_PRE_LABEL = "Peace Word Usage in"
PEACE_PRE_PATH = "war/peace_by_"
NORMALIZE_VALUE_PEACE = 20
Y_AXIS_LIMIT_PEACE = 1050

ECONOMY_PRE_LABEL = "Economy Word Usage in"
ECONOMY_PRE_PATH = "war/economy_by"
NORMALIZE_VALUE_ECONOMY = 25
Y_AXIS_LIMIT_ECONOMY = 800

TOPICS = ["PEACE", "VICTORY", "ECONOMY"]

"""
                  #############################
                  ## KEYWORD LISTS BY TOPICS ##
                  #############################
"""
VICTORY_WORDS = [
    "victory", "victories", "triumph", "triumphs", "triumphant",
    "win", "wins", "won", "winning", "prevail", "prevailed", "prevails", "prevailing",
    "conquer", "conquered", "conquers", "conquering",
    "dominate", "dominated", "dominates", "dominating",
    "defeat", "defeated", "defeating", "vanquish", "vanquished", "vanquishing",
    "rout", "routed", "unbeaten", "undefeated", "crush", "crushed", "crushing",
    "decisive", "overwhelm", "overwhelmed", "overwhelming",
    "supremacy", "superiority"
]

PEACE_WORDS = [
    "peace", "peaceful", "peacetime", "nonviolence", "nonviolent",
    "ceasefire", "armistice", "truce", "treaty", "accord", "agreement",
    "diplomacy", "diplomatic", "reconciliation", "reconcile",
    "resolution", "resolve", "negotiation", "negotiate", "talks",
    "settlement", "coexistence", "harmony", "understanding", "unity",
    "stability", "stabilize", "disarmament", "deescalation", "detente",
    "friendship", "alliance", "cooperation", "pact", "collaboration",
    "conciliate", "conciliation", "mediator", "mediation"
]

ECONOMY_WORDS = [
    # General economic terms in war contexts
    "economy", "economic", "economics", "economically",

    # Direct burdens and costs
    "cost", "costs", "burden", "burdens", "sacrifice", "sacrifices", "debt", "tax", "taxation",
    "inflation", "deficit", "expenditure", "spending", "expense", "expenses", "waste", "cuts",

    # Resource scarcity
    "shortage", "shortages", "ration", "rationing", "scarcity", "scarce",

    # Workforce & industry strain
    "labor", "labour", "employment", "unemployment", "jobs", "wages", "income", "productivity",
    "industry", "industries", "manufacturing", "output", "resources", "supplies",

    # Economic consequences
    "recession", "slowdown", "bankruptcy", "collapse", "unaffordable", "hardship", "poverty",
    "devaluation", "destabilize", "destabilization", "drain", "cripple", "crippling"
]

"""
                  ###############################################
                  ## KEYWORD LISTS TO IDENTIFY WAR-PEACE CYCLE ##
                  ###############################################
"""

WAR_MORAL_WORDS = [
    "victory", "freedom", "liberty", "duty", "resolve", "justice",
    "honor", "sacrifice", "courage", "defend", "defense", "fight",
    "enemy", "strength", "commitment", "valor", "righteous", "secure",
    "mission", "triumph", "stand firm", "determination", "rally",
    "protect", "destiny", "patriot", "noble cause", "champion",
    "endure", "unwavering", "march forward", "unyielding"
]

WAR_RESOLVE_WORDS = [
    "peace", "reconciliation", "harmony", "prosperity", "end hostilities",
    "withdrawal", "armistice", "truce", "ceasefire", "diplomacy", "negotiation",
    "settlement", "rebuild", "healing", "cooperation", "unity", "mutual respect",
    "reconstruction", "restoration", "partnership", "alliance", "friendship",
    "reunion", "nonviolence", "goodwill", "treaty", "accord", "understanding",
    "peaceful", "security", "transition", "renewal"
]

WAR_PERIODS = [
    ("Civil War", 1859, 1867),
    ("First World War", 1915, 1919),
    ("Second World War", 1937, 1947),
    ("Vietnam War", 1961, 1977),
    ("Iraq War\n&\nPost 9/11", 2001, 2013)
]

WARS_COMPARISON_CAPTION = ("Word count and frequency of topic-related terms in presidential speeches.\n"
                           "The plot highlights differences in terminology between World War II and the Vietnam War.")
WAR_AND_PEACE_CAPTION = ("Average use of fighting moral and peace terminology in presidential speeches.\n"
                         "The trends illustrate shifts related to war and peace: during wartime, fighting terms "
                         "dominate,"
                         "\nwhile in peacetime, peace terminology is more prevalent.")
