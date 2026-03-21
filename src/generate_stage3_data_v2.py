"""
Enhanced Stage 3 Dataset Generator v2
======================================
Improvements over v1:
1. Single-word glosses (HELLO, THANK-YOU, YES, NO, etc.)
2. More questions (20%+ instead of ~3.5%)
3. Dialogue sequences with context for conversational training
4. Paraphrase variations (multiple English outputs per gloss)
5. Longer sequences (7-10 tokens)
6. Better semantic validity (filtered invalid combinations)
7. Natural, conversational English output
"""

import pandas as pd
import random
import os
import inflect
from collections import defaultdict

p = inflect.engine()

# =====================================================================
# 1. VOCABULARY (Same as v1 but extended)
# =====================================================================
pronouns    = ["I", "YOU", "HE", "THEY", "WE", "SHE"]
people      = ["FRIEND", "BABY", "FATHER", "POLICE", "BOSS", "FAMILY", "WOMAN",
               "SISTER", "MAN", "TEAM", "BROTHER", "DOCTOR", "CHILD", "TEACHER",
               "MOTHER", "WORKER", "STUDENT", "GROUP"]
places      = ["COUNTRY", "ROOM", "PLACE", "BANK", "HOSPITAL", "CITY", "OFFICE",
               "SCHOOL", "HOUSE", "MARKET", "ROAD", "COMPANY", "STORE", "LIBRARY",
               "PARK", "RESTAURANT"]
tech        = ["PROGRAM", "MODEL", "INTERNET", "PHONE", "SYSTEM", "PASSWORD",
               "COMPUTER", "DATA", "CODE", "VIDEO", "CAMERA", "APP", "LAPTOP"]
consumables = ["EAT_FOOD", "WATER", "COFFEE", "APPLE", "MEDICINE"]
vehicles    = ["DRIVE_CAR", "BUS", "BIKE", "TRAIN"]
things      = ["LANGUAGE", "WORD", "PROJECT", "IDEA", "SOLUTION", "EXAM", "MONEY",
               "MEETING", "BOOK", "SENTENCE", "DESIGN", "NAME", "QUESTION",
               "PROBLEM", "PART", "KEY", "BAG", "LETTER", "SIGN", "LESSON"]
feelings    = ["ANGRY", "SORRY", "STRONG", "READY", "BUSY", "TIRED", "EXCITED",
               "HAPPY", "WEAK", "SAD", "COLD", "HOT", "SCARED", "CONFUSED",
               "BORED", "PROUD", "NERVOUS", "SICK"]
adj_size    = ["BIG", "SMALL", "SHORT", "LONG", "HIGH", "LOW", "TALL", "WIDE"]
adj_quality = ["GOOD", "BAD", "NEW", "OLD", "IMPORTANT", "DIFFERENT", "HARD_DIFFICULT",
               "SIMPLE", "FAST", "EASY", "CLEAR", "WRONG", "FULL",
               "EMPTY", "HEAVY", "LIGHT", "EXPENSIVE", "CHEAP"]
adj_people  = ["GOOD", "BAD", "IMPORTANT", "BUSY", "FREE", "OLD", "STRONG", "WEAK",
               "SMART", "KIND", "FAMOUS"]
times       = ["TODAY", "YESTERDAY", "TOMORROW", "NOW", "MORNING", "NIGHT", "AFTERNOON"]

mass_nouns  = consumables + ["MONEY", "DATA", "CODE", "SOFTWARE"]

# Merged token display names for natural English output
MERGED_TOKEN_ENGLISH = {
    "DRIVE_CAR": {"verb": "drive", "noun": "the car", "past": "drove", "ing": "driving"},
    "HARD_DIFFICULT": {"adj": "hard"},
    "MAKE_CREATE": {"verb": "make", "noun": "creation", "past": "made", "ing": "making"},
    "EAT_FOOD": {"verb": "eat", "noun": "some food", "past": "ate", "ing": "eating"},
}

def merged_token_lower(token):
    """Get a reasonable lowercase English form for merged tokens."""
    if token in MERGED_TOKEN_ENGLISH:
        info = MERGED_TOKEN_ENGLISH[token]
        return info.get("verb", info.get("adj", info.get("noun", token.lower())))
    return token.lower()
buyable     = tech + things + vehicles + consumables
all_nouns   = places + tech + things + vehicles + consumables

verbs_dict = {
    "eat":          ["EAT_FOOD", "COOK"],
    "drink":        ["DRINK"],
    "drive":        ["DRIVE_CAR"],
    "tech_digital": ["DOWNLOAD", "UPLOAD", "DELETE", "SAVE", "INSTALL"],
    "tech_general": ["USE", "FIX", "DEVELOP", "MAKE_CREATE", "PROGRAM", "SEND"],
    "visit":        ["VISIT", "GO", "COME", "WALK", "RUN", "CLIMB", "MOVE",
                     "LEAVE", "ARRIVE", "RETURN"],
    "study":        ["STUDY", "LEARN", "READ", "WRITE", "PRACTICE", "TEACH"],
    "mental":       ["KNOW", "UNDERSTAND", "REMEMBER", "FORGET", "THINK", "WANT",
                     "NEED", "LOVE", "LIKE", "MISS", "BELIEVE", "HOPE"],
    "buy_sell":     ["BUY", "SELL", "PAY", "GIVE", "TAKE", "RECEIVE", "BORROW", "BRING", "GET"],
    "help":         ["HELP", "CALL", "SHOW", "FIND", "MEET", "EXPLAIN"],
}

# =====================================================================
# 2. SEMANTIC VALIDITY RULES
# =====================================================================
INVALID_VERB_OBJECT = {
    'BUY': ['PASSWORD', 'NAME', 'IDEA', 'WORD', 'SENTENCE', 'LANGUAGE', 'DATA', 'CODE'],
    'SELL': ['PASSWORD', 'NAME', 'IDEA', 'WORD', 'SENTENCE', 'DATA', 'CODE'],
    'DRIVE_CAR': ['RESTAURANT', 'SCHOOL', 'HOSPITAL', 'LIBRARY', 'PARK', 'OFFICE'],
    'DRINK': ['EAT_FOOD', 'APPLE', 'BOOK', 'PHONE', 'COMPUTER'],
    'EAT_FOOD': ['WATER', 'COFFEE', 'PHONE', 'COMPUTER', 'DRIVE_CAR'],
    'UPLOAD': ['EAT_FOOD', 'WATER', 'DRIVE_CAR', 'HOUSE'],
    'DOWNLOAD': ['EAT_FOOD', 'WATER', 'DRIVE_CAR', 'HOUSE'],
}

def is_valid_combination(verb, obj):
    """Check if verb-object combination is semantically valid."""
    if verb in INVALID_VERB_OBJECT:
        return obj not in INVALID_VERB_OBJECT[verb]
    return True

# =====================================================================
# 3. HELPERS (Enhanced from v1)
# =====================================================================
def subject_english(s):
    if s == "I":   return "I", True, False
    if s == "YOU": return "you", False, True
    if s == "HE":  return "he", False, False
    if s == "SHE": return "she", False, False
    if s in ["THEY", "WE"]: return s.lower(), False, True
    if s in ["TEAM", "FAMILY", "GROUP", "POLICE"]: return f"the {s.lower()}", False, True
    return f"the {s.lower()}", False, False

def object_pronoun(s):
    return {"I": "me", "HE": "him", "SHE": "her", "WE": "us", "THEY": "them"}.get(s, s.lower())

def conjugate(verb, time_word, is_first, is_plural):
    past_tense = {
        "EAT": "ate", "DRINK": "drank", "DRIVE": "drove", "GO": "went", "COME": "came",
        "RUN": "ran", "READ": "read", "WRITE": "wrote", "BUY": "bought", "SELL": "sold",
        "PAY": "paid", "GIVE": "gave", "TAKE": "took", "LEAVE": "left", "BRING": "brought",
        "KNOW": "knew", "UNDERSTAND": "understood", "FORGET": "forgot", "THINK": "thought",
        "MEET": "met", "TEACH": "taught", "FIND": "found", "SEND": "sent",
        "SEE": "saw", "NEED": "needed", "STUDY": "studied", "PROGRAM": "programmed",
        "MISS": "missed", "BELIEVE": "believed", "HOPE": "hoped",
        "RECEIVE": "received", "BORROW": "borrowed", "ARRIVE": "arrived", "RETURN": "returned",
        "WALK": "walked", "CLIMB": "climbed", "MOVE": "moved", "VISIT": "visited",
        "LEARN": "learned", "PRACTICE": "practiced", "EXPLAIN": "explained",
        "CALL": "called", "SHOW": "showed", "DELETE": "deleted", "SAVE": "saved",
        "DOWNLOAD": "downloaded", "UPLOAD": "uploaded", "INSTALL": "installed",
        "DEVELOP": "developed", "MAKE": "made", "CREATE": "created", "USE": "used", "FIX": "fixed",
        "GET": "got", "HAVE": "had"
    }

    if verb == "HAVE":
        if time_word == "YESTERDAY": return "had"
        if time_word == "TOMORROW": return "will have"
        return "have" if (is_first or is_plural) else "has"

    # Handle merged tokens
    if verb == "DRIVE_CAR": verb = "DRIVE"
    if verb == "EAT_FOOD": verb = "EAT"
    if verb == "MAKE_CREATE": verb = "MAKE"

    if time_word == "YESTERDAY":
        if verb in past_tense: return past_tense[verb]
        if verb.endswith("E"): return f"{verb.lower()}d"
        if verb.endswith("Y"): return f"{verb.lower()[:-1]}ied"
        return f"{verb.lower()}ed"
    if time_word == "TOMORROW":
        return f"will {verb.lower()}"

    be = "am" if is_first else "are" if is_plural else "is"
    if verb in verbs_dict["mental"]:
        if is_first or is_plural:
            return verb.lower()
        base = verb.lower()
        if base.endswith(("ss", "sh", "ch", "x")):
            return base + "es"
        return base + "s"

    double_final = ["RUN", "SIT", "WIN", "CUT", "PROGRAM", "STOP", "GET"]
    if verb in double_final:
        ing = verb.lower() + verb.lower()[-1] + "ing"
    elif verb.endswith("E") and verb not in ["SEE", "FREE", "LOVE"]:
        ing = verb.lower()[:-1] + "ing"
    else:
        ing = verb.lower() + "ing"
    return f"{be} {ing}"

def format_time(t):
    return {"MORNING": "In the morning", "AFTERNOON": "In the afternoon",
            "NIGHT": "At night"}.get(t, t.capitalize())

def obj_article(noun):
    if noun in consumables:
        if noun == "EAT_FOOD": return "some food"
        return f"some {noun.lower()}"
    if noun == "DRIVE_CAR": return "the car"
    if noun == "HARD_DIFFICULT": return "hard"
    if noun == "MAKE_CREATE": return "creation"
    return f"the {noun.lower()}"

# =====================================================================
# 4. SINGLE-WORD GLOSSES (NEW!)
# =====================================================================
SINGLE_WORD_GLOSSES = [
    # Greetings
    ("HELLO", ["Hello.", "Hi.", "Hey.", "Hello!"]),
    ("GOODBYE", ["Goodbye.", "Bye.", "Bye-bye.", "See you!"]),
    ("HI", ["Hi.", "Hey.", "Hello."]),
    ("BYE", ["Bye.", "Goodbye.", "See ya."]),

    # Politeness
    ("THANK-YOU", ["Thank you.", "Thanks.", "Thanks a lot.", "Thank you very much."]),
    ("PLEASE", ["Please.", "Please?"]),
    ("SORRY", ["Sorry.", "I'm sorry.", "My apologies."]),
    ("EXCUSE-ME", ["Excuse me.", "Pardon me."]),

    # Yes/No
    ("YES", ["Yes.", "Yeah.", "Yep.", "Yes!"]),
    ("NO", ["No.", "Nope.", "No way."]),
    ("MAYBE", ["Maybe.", "Perhaps.", "Possibly."]),
    ("OK", ["Okay.", "OK.", "Alright.", "Sure."]),

    # Questions (single word)
    ("WHAT", ["What?", "What is it?"]),
    ("WHY", ["Why?", "Why is that?"]),
    ("HOW", ["How?", "How so?"]),
    ("WHERE", ["Where?", "Where is it?"]),
    ("WHO", ["Who?", "Who is it?"]),
    ("WHEN", ["When?", "When is it?"]),

    # Common responses
    ("WAIT", ["Wait.", "Hold on.", "Just a moment."]),
    ("STOP", ["Stop.", "Stop it."]),
    ("HELP", ["Help!", "Help me!"]),
    ("AGAIN", ["Again.", "One more time.", "Repeat that."]),
    ("FINISH", ["Done.", "Finished.", "I'm done."]),
    ("UNDERSTAND", ["I understand.", "Got it.", "Understood."]),
    ("RIGHT", ["Right.", "Correct.", "That's right."]),
    ("WRONG", ["Wrong.", "Incorrect.", "That's wrong."]),

    # Feelings (single word)
    ("HAPPY", ["Happy.", "I'm happy."]),
    ("SAD", ["Sad.", "I'm sad."]),
    ("TIRED", ["Tired.", "I'm tired."]),
    ("HUNGRY", ["Hungry.", "I'm hungry."]),
    ("GOOD", ["Good.", "I'm good."]),
    ("BAD", ["Bad.", "Not good."]),
]

def generate_single_word_glosses():
    """Generate single-word gloss pairs with variations."""
    rows = []
    for gloss, texts in SINGLE_WORD_GLOSSES:
        for text in texts:
            rows.append({"gloss": gloss, "text": text})
    # Repeat to give proper weight
    return rows * 100

# =====================================================================
# 5. DIALOGUE SEQUENCES (NEW!)
# =====================================================================
DIALOGUE_TEMPLATES = [
    # Greeting sequences
    {
        'turns': [
            ('A', 'HELLO', 'Hello!'),
            ('B', 'HELLO HOW YOU', 'Hi! How are you?'),
            ('A', 'I GOOD THANK-YOU', "I'm good, thanks."),
            ('B', 'GOOD', "Good to hear!"),
        ]
    },
    # Name introduction
    {
        'turns': [
            ('A', 'NAME WHAT YOU', "What's your name?"),
            ('B', 'MY NAME JOHN', "My name is John."),
            ('A', 'NICE MEET YOU', 'Nice to meet you!'),
            ('B', 'NICE MEET YOU', 'Nice to meet you too!'),
        ]
    },
    # Location questions
    {
        'turns': [
            ('A', 'WHERE YOU GO', 'Where are you going?'),
            ('B', 'I GO STORE', "I'm going to the store."),
            ('A', 'WHY', 'Why?'),
            ('B', 'I NEED FOOD', "I need some food."),
            ('A', 'OK', 'Okay.'),
        ]
    },
    # Help request
    {
        'turns': [
            ('A', 'PLEASE HELP ME', 'Can you help me, please?'),
            ('B', 'YES WHAT YOU NEED', 'Sure, what do you need?'),
            ('A', 'I NEED MONEY', "I need some money."),
            ('B', 'OK I HELP', "Okay, I'll help."),
        ]
    },
    # Clarification
    {
        'turns': [
            ('A', 'I NOT UNDERSTAND', "I don't understand."),
            ('B', 'SORRY I SIGN AGAIN', "Sorry, let me sign again."),
            ('A', 'OK THANK-YOU', 'Okay, thank you.'),
        ]
    },
    # Farewell
    {
        'turns': [
            ('A', 'I GO NOW', "I have to go now."),
            ('B', 'OK SEE YOU LATER', 'Okay, see you later!'),
            ('A', 'BYE', 'Bye!'),
            ('B', 'BYE', 'Bye, take care!'),
        ]
    },
    # Feeling inquiry
    {
        'turns': [
            ('A', 'YOU OK YOU-KNOW', 'Are you okay?'),
            ('B', 'YES I FINE', "Yes, I'm fine."),
            ('A', 'GOOD', 'Good.'),
        ]
    },
    # Time questions
    {
        'turns': [
            ('A', 'WHEN YOU COME', 'When are you coming?'),
            ('B', 'TOMORROW', "Tomorrow."),
            ('A', 'OK SEE YOU TOMORROW', 'Okay, see you tomorrow!'),
        ]
    },
]

NAMES = ['JOHN', 'MARIA', 'ALEX', 'SARAH', 'CHRIS', 'EMMA']
DIALOGUE_PLACES = ['STORE', 'SCHOOL', 'LIBRARY', 'HOSPITAL', 'PARK', 'OFFICE', 'HOME', 'RESTAURANT']
DIALOGUE_OBJECTS = ['BOOK', 'PHONE', 'COMPUTER', 'FOOD', 'WATER', 'HELP', 'MONEY', 'MEDICINE']

def generate_dialogue_dataset(num_dialogues=3000):
    """Generate conversational dialogue pairs with context."""
    dialogues = []
    dialogue_id = 0

    for _ in range(num_dialogues):
        template = random.choice(DIALOGUE_TEMPLATES)
        dialogue_id += 1

        context_parts = []
        for turn_num, (speaker, gloss_template, text_template) in enumerate(template['turns'], 1):
            gloss = gloss_template
            text = text_template

            # Fill in placeholders
            if 'JOHN' in gloss:
                name = random.choice(NAMES)
                gloss = gloss.replace('JOHN', name)
                text = text.replace('John', name.capitalize())

            # Create context string from previous turns
            context_str = ' | '.join(context_parts[-2:]) if context_parts else ''

            dialogues.append({
                'dialogue_id': dialogue_id,
                'turn_number': turn_num,
                'speaker': speaker,
                'gloss': gloss,
                'text': text,
                'context': context_str
            })

            context_parts.append(f"{speaker}: {text}")

    return dialogues

# =====================================================================
# 6. ENHANCED QUESTION GENERATORS (More variety!)
# =====================================================================
def gen_wh_question_enhanced():
    """Enhanced WH-question generator with more variety."""
    templates = [
        # What questions
        ("what_doing", lambda: (
            f"WHAT {random.choice(pronouns)} DO",
            f"What {'am I' if random.choice(pronouns) == 'I' else 'are you' if random.choice(pronouns) == 'YOU' else 'is ' + subject_english(random.choice(pronouns))[0]} doing?"
        )),
        ("what_name", lambda: (
            f"NAME WHAT YOU",
            "What is your name?"
        )),
        ("what_want", lambda: (
            f"{(s := random.choice(pronouns))} WANT WHAT",
            f"What {'do' if s in ['I', 'YOU', 'WE', 'THEY'] else 'does'} {subject_english(s)[0]} want?"
        )),
        ("what_need", lambda: (
            f"{(s := random.choice(pronouns))} NEED WHAT",
            f"What {'do' if s in ['I', 'YOU', 'WE', 'THEY'] else 'does'} {subject_english(s)[0]} need?"
        )),
        ("what_time", lambda: (
            "TIME WHAT",
            "What time is it?"
        )),

        # Where questions
        ("where_go", lambda: (
            f"WHERE {random.choice(pronouns)} GO",
            f"Where {'am I' if random.choice(pronouns) == 'I' else 'are you' if random.choice(pronouns) == 'YOU' else 'is ' + subject_english(random.choice(pronouns))[0]} going?"
        )),
        ("where_object", lambda: (
            f"{(o := random.choice(things + tech))} WHERE",
            f"Where is the {o.lower()}?"
        )),
        ("where_live", lambda: (
            f"{(s := random.choice(pronouns))} LIVE WHERE",
            f"Where {'do' if s in ['I', 'YOU', 'WE', 'THEY'] else 'does'} {subject_english(s)[0]} live?"
        )),

        # Who questions
        ("who_person", lambda: (
            f"WHO {(role := random.choice(['TEACHER', 'DOCTOR', 'BOSS', 'FRIEND']))}",
            f"Who is the {role.lower()}?"
        )),
        ("who_come", lambda: (
            "WHO COME",
            "Who is coming?"
        )),

        # When questions
        ("when_event", lambda: (
            f"WHEN {(s := random.choice(pronouns))} {random.choice(['GO', 'COME', 'LEAVE', 'ARRIVE'])}",
            f"When {'am I' if s == 'I' else 'are you' if s == 'YOU' else 'is ' + subject_english(s)[0]} {random.choice(['going', 'coming', 'leaving', 'arriving'])}?"
        )),
        ("when_start", lambda: (
            f"WHEN {(o := random.choice(['MEETING', 'LESSON', 'EXAM', 'PROJECT']))} START",
            f"When does the {o.lower()} start?"
        )),

        # Why questions
        ("why_feel", lambda: (
            f"WHY {(s := random.choice(pronouns))} FEEL {(f := random.choice(feelings))}",
            f"Why {'am I' if s == 'I' else 'are you' if s == 'YOU' else 'is ' + subject_english(s)[0]} feeling {f.lower()}?"
        )),
        ("why_do", lambda: (
            f"WHY {(s := random.choice(pronouns))} {(v := random.choice(['GO', 'LEAVE', 'COME', 'WANT']))}",
            f"Why {'am I' if s == 'I' else 'are you' if s == 'YOU' else 'is ' + subject_english(s)[0]} {v.lower()}ing?"
        )),

        # How questions
        ("how_feel", lambda: (
            f"HOW {(s := random.choice(pronouns))} FEEL",
            f"How {'do I' if s == 'I' else 'do you' if s == 'YOU' else 'does ' + subject_english(s)[0]} feel?"
        )),
        ("how_many", lambda: (
            f"HOW-MANY {(o := random.choice(things + tech))} {(s := random.choice(pronouns))} HAVE",
            f"How many {p.plural(o.lower())} {'do I' if s == 'I' else 'do you' if s == 'YOU' else 'does ' + subject_english(s)[0]} have?"
        )),
        ("how_much", lambda: (
            f"HOW-MUCH {(o := random.choice(mass_nouns))} {(s := random.choice(pronouns))} NEED",
            f"How much {o.lower()} {'do I' if s == 'I' else 'do you' if s == 'YOU' else 'does ' + subject_english(s)[0]} need?"
        )),
    ]

    template_name, gen_fn = random.choice(templates)
    try:
        return gen_fn()
    except:
        return "WHAT YOU WANT", "What do you want?"


def gen_yn_question_enhanced():
    """Enhanced Yes/No question generator."""
    s = random.choice(pronouns)
    s_eng, is_first, is_plural = subject_english(s)
    be = "am" if is_first else "are" if is_plural else "is"
    do = "do" if (is_first or is_plural) else "does"

    templates = [
        # Feeling questions
        lambda: (f"{s} FEEL {(f := random.choice(feelings))} YOU-KNOW",
                 f"{'Am I' if s == 'I' else 'Are you' if s == 'YOU' else be.capitalize() + ' ' + s_eng} feeling {f.lower()}?"),

        # Have questions
        lambda: (f"{s} HAVE {(o := random.choice(things + tech))} YOU-KNOW",
                 f"{'Do I' if s == 'I' else 'Do you' if s == 'YOU' else do.capitalize() + ' ' + s_eng} have the {o.lower()}?"),

        # Going questions
        lambda: (f"{s} GO {(place := random.choice(places))} YOU-KNOW",
                 f"{'Am I' if s == 'I' else 'Are you' if s == 'YOU' else be.capitalize() + ' ' + s_eng} going to the {place.lower()}?"),

        # Want questions
        lambda: (f"{s} WANT {(o := random.choice(things + consumables))} YOU-KNOW",
                 f"{'Do I' if s == 'I' else 'Do you' if s == 'YOU' else do.capitalize() + ' ' + s_eng} want {obj_article(o)}?"),

        # Need questions
        lambda: (f"{s} NEED HELP YOU-KNOW",
                 f"{'Do I' if s == 'I' else 'Do you' if s == 'YOU' else do.capitalize() + ' ' + s_eng} need help?"),

        # Like questions
        lambda: (f"{s} LIKE {(o := random.choice(things + places + consumables))} YOU-KNOW",
                 f"{'Do I' if s == 'I' else 'Do you' if s == 'YOU' else do.capitalize() + ' ' + s_eng} like {obj_article(o)}?"),

        # OK questions
        lambda: (f"{s} OK YOU-KNOW",
                 f"{'Am I' if s == 'I' else 'Are you' if s == 'YOU' else be.capitalize() + ' ' + s_eng} okay?"),

        # Ready questions
        lambda: (f"{s} READY YOU-KNOW",
                 f"{'Am I' if s == 'I' else 'Are you' if s == 'YOU' else be.capitalize() + ' ' + s_eng} ready?"),
    ]

    return random.choice(templates)()

# =====================================================================
# 7. PARAPHRASE VARIATIONS (NEW!)
# =====================================================================
PARAPHRASE_TEMPLATES = {
    "HELLO HOW YOU": [
        "Hello, how are you?",
        "Hi, how are you doing?",
        "Hey, how's it going?",
        "Hello! How are you?",
    ],
    "I GO STORE": [
        "I am going to the store.",
        "I'm going to the store.",
        "I'm heading to the store.",
        "I will go to the store.",
    ],
    "THANK-YOU": [
        "Thank you.",
        "Thanks.",
        "Thank you very much.",
        "Thanks a lot.",
    ],
    "I NEED HELP": [
        "I need help.",
        "I need some help.",
        "Can you help me?",
        "I need assistance.",
    ],
    "WHERE YOU GO": [
        "Where are you going?",
        "Where are you headed?",
        "Where are you off to?",
    ],
    "I NOT UNDERSTAND": [
        "I don't understand.",
        "I do not understand.",
        "I'm confused.",
        "I didn't get that.",
    ],
    "SEE YOU TOMORROW": [
        "See you tomorrow.",
        "See you tomorrow!",
        "I'll see you tomorrow.",
        "Until tomorrow!",
    ],
    "NICE MEET YOU": [
        "Nice to meet you.",
        "Nice to meet you!",
        "Pleased to meet you.",
        "It's nice to meet you.",
    ],
}

def generate_paraphrases():
    """Generate paraphrase variations for common phrases."""
    rows = []
    for gloss, texts in PARAPHRASE_TEMPLATES.items():
        for text in texts:
            rows.append({"gloss": gloss, "text": text})
    return rows * 50  # Give weight


# =====================================================================
# 8. LONGER SEQUENCES (7-10 tokens)
# =====================================================================
def gen_long_sequence():
    """Generate longer, more complex sentences (7-10 tokens)."""
    templates = [
        # Compound time sentences
        lambda: (
            f"YESTERDAY I GO {(p1 := random.choice(places))} {(v := random.choice(['BUY', 'SEE', 'FIND', 'GET']))} {(o := random.choice(buyable))} COME HOME",
            f"Yesterday, I went to the {p1.lower()}, {conjugate(v, 'YESTERDAY', True, False)} {obj_article(o)}, and came home."
        ),

        # Future plans
        lambda: (
            f"TOMORROW MORNING I WANT GO {(p1 := random.choice(places))} {(v := random.choice(['STUDY', 'READ', 'LEARN']))} {(o := random.choice(things))}",
            f"Tomorrow morning, I want to go to the {p1.lower()} and {v.lower()} the {o.lower()}."
        ),

        # Past with friend
        lambda: (
            f"YESTERDAY MY FRIEND VISIT MY HOUSE WE EAT {(f := random.choice(consumables))}",
            f"Yesterday, my friend visited my house and we ate {f.lower() if f in mass_nouns else obj_article(f)}."
        ),

        # Sequential actions
        lambda: (
            f"TODAY I WAKE UP EAT {(f := random.choice(['FOOD', 'APPLE']))} GO {(p := random.choice(places))}",
            f"Today, I woke up, ate {obj_article(f)}, and went to the {p.lower()}."
        ),

        # Request with reason
        lambda: (
            f"PLEASE HELP ME I NEED {(o := random.choice(things + tech))} FOR {(p := random.choice(things))}",
            f"Please help me, I need the {o.lower()} for the {p.lower()}."
        ),
    ]

    return random.choice(templates)()


# =====================================================================
# 9. STANDARD GENERATORS (from v1, with validity filtering)
# =====================================================================
def gen_time_svo():
    t = random.choice(times)
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)

    cat = random.choice(list(verbs_dict.keys()))
    v = random.choice(verbs_dict[cat])

    # Get object with validity check
    for _ in range(10):  # Max attempts
        if cat == "eat": o = random.choice(["FOOD", "APPLE"])
        elif cat == "drink": o = random.choice(["WATER", "COFFEE", "MEDICINE"])
        elif cat == "drive": o = random.choice(vehicles)
        elif cat == "tech_digital": o = random.choice(tech)
        elif cat == "tech_general": o = random.choice(tech + things)
        elif cat == "visit": o = random.choice(places)
        elif cat == "study": o = random.choice(things + tech)
        elif cat == "help": o = random.choice(pronouns + people)
        elif cat == "mental": o = random.choice(things + places)
        else: o = random.choice(buyable)

        if is_valid_combination(v, o):
            break

    if cat == "eat": o_eng = obj_article(o)
    elif cat == "drink": o_eng = f"some {o.lower()}"
    elif cat == "drive": o_eng = f"the {o.lower()}"
    elif cat == "tech_digital": o_eng = f"the {o.lower()}"
    elif cat == "tech_general": o_eng = f"the {o.lower()}"
    elif cat == "visit":
        if v == "ARRIVE": prep = "at "
        elif v in ["GO","WALK","RUN","COME","MOVE","CLIMB","RETURN"]: prep = "to "
        else: prep = ""
        o_eng = f"{prep}the {o.lower()}"
    elif cat == "study": o_eng = f"the {o.lower()}"
    elif cat == "help":
        o_eng = object_pronoun(o) if o in pronouns else f"the {o.lower()}"
    elif cat == "mental": o_eng = f"the {o.lower()}"
    else: o_eng = obj_article(o)

    v_eng = conjugate(v, t, is_first, is_plural)
    return f"{t} {s} {v} {o}", f"{format_time(t)}, {s_eng} {v_eng} {o_eng}."


def gen_negation():
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)
    v = random.choice(["WANT", "NEED", "LIKE", "HAVE", "KNOW", "UNDERSTAND", "SEE"])

    o = random.choice(things + tech + places)
    do_aux = "do" if (is_first or is_plural) else "does"
    v_base = "have" if v == "HAVE" else v.lower()
    return f"{s} NOT {v} {o}", f"{s_eng.capitalize()} {do_aux} not {v_base} {obj_article(o)}."


def gen_feeling():
    t = random.choice(times)
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)
    o = random.choice(feelings)
    be = "am" if is_first else "are" if is_plural else "is"
    if t == "YESTERDAY": v_eng = "felt"
    elif t == "TOMORROW": v_eng = "will feel"
    else: v_eng = f"{be} feeling"
    return f"{t} {s} FEEL {o}", f"{format_time(t)}, {s_eng} {v_eng} {o.lower()}."


def gen_imperative():
    templates = [
        ("PLEASE WAIT", "Please wait."),
        ("PLEASE HELP", "Please help."),
        ("PLEASE COME", "Please come."),
        ("PLEASE STOP", "Please stop."),
        ("PLEASE LISTEN", "Please listen."),
        ("PLEASE SIT", "Please sit down."),
        ("HELP ME PLEASE", "Please help me."),
        ("COME HERE", "Come here."),
        ("STOP THAT", "Stop that."),
        (f"PLEASE BRING {(o := random.choice(things))}", f"Please bring the {o.lower()}."),
        (f"PLEASE SHOW ME {(o := random.choice(things + tech))}", f"Please show me the {o.lower()}."),
        (f"PLEASE GIVE ME {(o := random.choice(things))}", f"Please give me the {o.lower()}."),
    ]
    return random.choice(templates)


# =====================================================================
# 9b. MERGED-LABEL DISAMBIGUATION DATA
# =====================================================================
MERGED_DISAMBIGUATION = [
    # DRIVE_CAR: context determines "drive" (verb) vs "car" (noun)
    ("I DRIVE_CAR STORE", "I'm driving to the store."),
    ("I DRIVE_CAR SCHOOL", "I'm driving to school."),
    ("I DRIVE_CAR HOME", "I'm driving home."),
    ("DRIVE_CAR WHERE", "Where is the car?"),
    ("MY DRIVE_CAR NEW", "My car is new."),
    ("MY DRIVE_CAR OLD", "My car is old."),
    ("I LIKE DRIVE_CAR", "I like driving."),
    ("I NEED DRIVE_CAR", "I need a car."),
    ("YESTERDAY I DRIVE_CAR HOSPITAL", "Yesterday I drove to the hospital."),
    ("TOMORROW I DRIVE_CAR WORK", "Tomorrow I will drive to work."),
    ("DRIVE_CAR FAST", "The car is fast."),
    ("I BUY DRIVE_CAR", "I'm buying a car."),
    ("HE DRIVE_CAR BUS", "He drives a bus."),

    # HARD_DIFFICULT: always adjective, context varies
    ("EXAM HARD_DIFFICULT", "The exam is hard."),
    ("WORK HARD_DIFFICULT", "The work is difficult."),
    ("THIS HARD_DIFFICULT", "This is hard."),
    ("VERY HARD_DIFFICULT", "It's very difficult."),
    ("NOT HARD_DIFFICULT", "It's not hard."),
    ("LIFE HARD_DIFFICULT", "Life is hard."),
    ("QUESTION HARD_DIFFICULT", "The question is difficult."),
    ("I THINK HARD_DIFFICULT", "I think it's hard."),

    # MAKE_CREATE: context determines "make" vs "create"
    ("I MAKE_CREATE EAT_FOOD", "I'm making food."),
    ("I MAKE_CREATE PROJECT", "I'm creating a project."),
    ("I MAKE_CREATE DESIGN", "I'm creating a design."),
    ("PLEASE MAKE_CREATE COFFEE", "Please make some coffee."),
    ("HE MAKE_CREATE PROGRAM", "He's creating a program."),
    ("WE MAKE_CREATE PLAN", "We're making a plan."),
    ("I WANT MAKE_CREATE", "I want to make it."),
    ("I NEED MAKE_CREATE", "I need to create it."),
    ("YESTERDAY I MAKE_CREATE", "Yesterday I made it."),
    ("TOMORROW I MAKE_CREATE", "Tomorrow I will make it."),

    # EAT_FOOD: context determines "eat" (verb) vs "food" (noun)
    ("I EAT_FOOD", "I'm eating."),
    ("I EAT_FOOD NOW", "I'm eating now."),
    ("I NEED EAT_FOOD", "I need some food."),
    ("WHERE EAT_FOOD", "Where is the food?"),
    ("EAT_FOOD GOOD", "The food is good."),
    ("EAT_FOOD BAD", "The food is bad."),
    ("I LIKE EAT_FOOD", "I like the food."),
    ("I WANT EAT_FOOD", "I want to eat."),
    ("YESTERDAY I EAT_FOOD", "Yesterday I ate."),
    ("TOMORROW I EAT_FOOD", "Tomorrow I will eat."),
    ("I BUY EAT_FOOD", "I'm buying food."),
    ("WE EAT_FOOD TOGETHER", "We're eating together."),
    ("I HUNGRY EAT_FOOD PLEASE", "I'm hungry, food please."),

    # Composite labels that should stay as-is
    ("ALSO_SAME I THINK", "I think so too."),
    ("ALSO_SAME", "Same here."),
    ("HE_SHE GO STORE", "They're going to the store."),
    ("I_ME WANT HELP", "I want help."),
    ("US_WE GO TOGETHER", "We're going together."),
    ("MARKET_STORE WHERE", "Where is the store?"),
    ("FEW_SEVERAL PEOPLE COME", "A few people are coming."),
    ("HIS_HER BOOK WHERE", "Where is their book?"),
]

def generate_merged_disambiguation():
    """Generate disambiguation training data for merged labels."""
    rows = []
    for gloss, text in MERGED_DISAMBIGUATION:
        rows.append({"gloss": gloss, "text": text})
    return rows * 120  # High weight for these critical patterns

# =====================================================================
# 10. MAIN GENERATION
# =====================================================================
def main():
    print("=" * 60)
    print("Enhanced Stage 3 Dataset Generator v2")
    print("=" * 60)

    all_data = []

    # 1. Single-word glosses (important for edge cases)
    print("\n1. Generating single-word glosses...")
    single_word = generate_single_word_glosses()
    all_data.extend(single_word)
    print(f"   Generated {len(single_word)} single-word pairs")

    # 1b. Merged-label disambiguation data
    print("\n1b. Generating merged-label disambiguation data...")
    merged_data = generate_merged_disambiguation()
    all_data.extend(merged_data)
    print(f"   Generated {len(merged_data)} disambiguation pairs")

    # 2. Dialogues with context
    print("\n2. Generating dialogue sequences...")
    dialogues = generate_dialogue_dataset(3000)
    # Convert to simple format (without context for standard training)
    dialogue_simple = [{"gloss": d["gloss"], "text": d["text"]} for d in dialogues]
    all_data.extend(dialogue_simple)
    print(f"   Generated {len(dialogue_simple)} dialogue pairs")

    # Save dialogue data separately for conversational training
    df_dialogues = pd.DataFrame(dialogues)
    df_dialogues.to_csv("slt_dialogue_dataset.csv", index=False)
    print(f"   Saved dialogue dataset to slt_dialogue_dataset.csv")

    # 3. Paraphrase variations
    print("\n3. Generating paraphrase variations...")
    paraphrases = generate_paraphrases()
    all_data.extend(paraphrases)
    print(f"   Generated {len(paraphrases)} paraphrase pairs")

    # 4. Questions (target: 20% of total)
    print("\n4. Generating questions (enhanced)...")
    question_count = 8000
    questions = []
    for _ in range(question_count):
        if random.random() < 0.6:
            try:
                g, e = gen_wh_question_enhanced()
                questions.append({"gloss": g, "text": e})
            except:
                pass
        else:
            try:
                g, e = gen_yn_question_enhanced()
                questions.append({"gloss": g, "text": e})
            except:
                pass
    all_data.extend(questions)
    print(f"   Generated {len(questions)} question pairs")

    # 5. Longer sequences
    print("\n5. Generating longer sequences...")
    long_seqs = []
    for _ in range(3000):
        try:
            g, e = gen_long_sequence()
            long_seqs.append({"gloss": g, "text": e})
        except:
            pass
    all_data.extend(long_seqs)
    print(f"   Generated {len(long_seqs)} long sequence pairs")

    # 6. Standard generators
    print("\n6. Generating standard sentences...")
    generators = [
        (gen_time_svo, 15),
        (gen_negation, 8),
        (gen_feeling, 5),
        (gen_imperative, 8),
    ]

    pool = []
    for fn, weight in generators:
        pool.extend([fn] * weight)

    for _ in range(20000):
        fn = random.choice(pool)
        try:
            result = fn()
            if isinstance(result, tuple):
                g, e = result
            else:
                continue
            all_data.append({"gloss": g, "text": e})
        except:
            pass

    print(f"   Generated ~20000 standard pairs")

    # Create DataFrame and deduplicate
    print("\n7. Processing and deduplicating...")
    df = pd.DataFrame(all_data)
    df = df.dropna().drop_duplicates(subset=["gloss"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate question percentage
    question_words = ["WHAT", "WHERE", "WHO", "WHEN", "WHY", "HOW", "YOU-KNOW"]
    is_question = df["gloss"].apply(lambda x: any(w in x for w in question_words))
    question_pct = is_question.sum() / len(df) * 100

    # Calculate single-word percentage
    is_single = df["gloss"].apply(lambda x: len(x.split()) == 1)
    single_pct = is_single.sum() / len(df) * 100

    # Calculate average gloss length
    avg_len = df["gloss"].apply(lambda x: len(x.split())).mean()

    print(f"\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Question percentage: {question_pct:.1f}%")
    print(f"Single-word percentage: {single_pct:.1f}%")
    print(f"Average gloss length: {avg_len:.2f} tokens")
    print(f"Unique glosses: {df['gloss'].nunique()}")
    print(f"Unique texts: {df['text'].nunique()}")

    # Save
    save_path = "slt_stage3_dataset_v2.csv"
    df.to_csv(save_path, index=False)
    print(f"\n✅ Saved {len(df)} total pairs to {save_path}")

    # Show sample
    print("\nSample rows:")
    for _, row in df.sample(10, random_state=42).iterrows():
        print(f"  [{row['gloss']}] → {row['text']}")


if __name__ == "__main__":
    main()
