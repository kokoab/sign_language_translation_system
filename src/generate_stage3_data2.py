import pandas as pd
import random
import os
import inflect

p = inflect.engine()

# =====================================================================
# 1. EXTRACT CLASSES FROM FOLDER
# =====================================================================
LANDMARKS_DIR = "ASL_landmarks_float16/"
try:
    files = [f for f in os.listdir(LANDMARKS_DIR) if f.endswith('.npy')]
    all_words = sorted(list(set([f.split('_')[0].upper() for f in files])))
except (FileNotFoundError, PermissionError) as e:
    print(f"⚠️ Warning: {e}. Continuing with empty all_words.")
    all_words = []

# =====================================================================
# 2. VOCABULARY
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
consumables = ["FOOD", "WATER", "COFFEE", "APPLE", "MEDICINE"]   
vehicles    = ["CAR", "BUS", "BIKE", "TRAIN"]
things      = ["LANGUAGE", "WORD", "PROJECT", "IDEA", "SOLUTION", "EXAM", "MONEY",
               "MEETING", "BOOK", "SENTENCE", "DESIGN", "NAME", "QUESTION",
               "PROBLEM", "PART", "KEY", "BAG", "LETTER", "SIGN", "LESSON"]
feelings    = ["ANGRY", "SORRY", "STRONG", "READY", "BUSY", "TIRED", "EXCITED",
               "HAPPY", "WEAK", "SAD", "COLD", "HOT", "SCARED", "CONFUSED",
               "BORED", "PROUD", "NERVOUS", "SICK"]
adj_size    = ["BIG", "SMALL", "SHORT", "LONG", "HIGH", "LOW", "TALL", "WIDE"]
adj_quality = ["GOOD", "BAD", "NEW", "OLD", "IMPORTANT", "DIFFERENT", "HARD",
               "SIMPLE", "FAST", "DIFFICULT", "EASY", "CLEAR", "WRONG", "FULL",
               "EMPTY", "HEAVY", "LIGHT", "EXPENSIVE", "CHEAP"]
adj_people  = ["GOOD", "BAD", "IMPORTANT", "BUSY", "FREE", "OLD", "STRONG", "WEAK",
               "SMART", "KIND", "FAMOUS"]
times       = ["TODAY", "YESTERDAY", "TOMORROW", "NOW", "MORNING", "NIGHT", "AFTERNOON"]

mass_nouns  = consumables + ["MONEY", "DATA", "CODE", "SOFTWARE"]

verbs_dict = {
    "eat":          ["EAT", "COOK"],
    "drink":        ["DRINK"],
    "drive":        ["DRIVE"],
    "tech_digital": ["DOWNLOAD", "UPLOAD", "DELETE", "SAVE", "INSTALL"],  
    "tech_general": ["USE", "FIX", "DEVELOP", "CREATE", "PROGRAM", "SEND"],  
    "visit":        ["VISIT", "GO", "COME", "WALK", "RUN", "CLIMB", "MOVE",
                     "LEAVE", "ARRIVE", "RETURN"],
    "study":        ["STUDY", "LEARN", "READ", "WRITE", "PRACTICE", "TEACH"],
    "mental":       ["KNOW", "UNDERSTAND", "REMEMBER", "FORGET", "THINK", "WANT",
                     "NEED", "LOVE", "LIKE", "MISS", "BELIEVE", "HOPE"],
    "buy_sell":     ["BUY", "SELL", "PAY", "GIVE", "TAKE", "RECEIVE", "BORROW", "BRING", "GET"],
    "help":         ["HELP", "CALL", "SHOW", "FIND", "MEET", "EXPLAIN"],
}

buyable   = tech + things + vehicles + consumables
all_nouns = places + tech + things + vehicles + consumables

# =====================================================================
# 3. HELPERS
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
    """FIX: Maps nominative pronouns to accusative when they act as objects."""
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
        "DEVELOP": "developed", "CREATE": "created", "USE": "used", "FIX": "fixed",
        "GET": "got", "HAVE": "had"
    }
    
    if verb == "HAVE":
        if time_word == "YESTERDAY": return "had"
        if time_word == "TOMORROW": return "will have"
        return "have" if (is_first or is_plural) else "has"
        
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
    if noun in consumables: return f"some {noun.lower()}"
    return f"the {noun.lower()}"

# =====================================================================
# 4. SENTENCE GENERATORS
# =====================================================================

def gen_time_svo():
    t = random.choice(times)
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)

    cat = random.choice(list(verbs_dict.keys()))
    v = random.choice(verbs_dict[cat])

    if cat == "eat": o = random.choice(["FOOD", "APPLE"]); o_eng = obj_article(o)
    elif cat == "drink": o = random.choice(["WATER", "COFFEE", "MEDICINE"]); o_eng = f"some {o.lower()}"
    elif cat == "drive": o = random.choice(vehicles); o_eng = f"the {o.lower()}"
    elif cat == "tech_digital": o = random.choice(tech); o_eng = f"the {o.lower()}"
    elif cat == "tech_general": o = random.choice(tech + things); o_eng = f"the {o.lower()}"
    elif cat == "visit":
        o = random.choice(places)
        # FIX: Correct prepositions for Arrive vs Leave vs Motion verbs
        if v == "ARRIVE": prep = "at "
        elif v in ["GO","WALK","RUN","COME","MOVE","CLIMB","RETURN"]: prep = "to "
        else: prep = ""
        o_eng = f"{prep}the {o.lower()}"
    elif cat == "study": o = random.choice(things + tech); o_eng = f"the {o.lower()}"
    elif cat == "help": 
        o = random.choice(pronouns + people)
        # FIX: Accusative pronouns for objects
        o_eng = object_pronoun(o) if o in pronouns else f"the {o.lower()}"
    elif cat == "mental": o = random.choice(things + places); o_eng = f"the {o.lower()}"
    else: o = random.choice(buyable); o_eng = obj_article(o)

    v_eng = conjugate(v, t, is_first, is_plural)
    return f"{t} {s} {v} {o}", f"{format_time(t)}, {s_eng} {v_eng} {o_eng}."

def gen_no_time_svo():
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)
    cat = random.choice(list(verbs_dict.keys()))
    v = random.choice(verbs_dict[cat])

    if cat == "eat": o = random.choice(["FOOD", "APPLE"]); o_eng = obj_article(o)
    elif cat == "drink": o = random.choice(["WATER", "COFFEE", "MEDICINE"]); o_eng = f"some {o.lower()}"
    elif cat == "drive": o = random.choice(vehicles); o_eng = f"the {o.lower()}"
    elif cat == "tech_digital": o = random.choice(tech); o_eng = f"the {o.lower()}"
    elif cat == "tech_general": o = random.choice(tech + things); o_eng = f"the {o.lower()}"
    elif cat == "visit":
        o = random.choice(places)
        # FIX: Correct prepositions for Arrive vs Leave vs Motion verbs
        if v == "ARRIVE": prep = "at "
        elif v in ["GO","WALK","RUN","COME","MOVE","CLIMB","RETURN"]: prep = "to "
        else: prep = ""
        o_eng = f"{prep}the {o.lower()}"
    elif cat == "study": o = random.choice(things + tech); o_eng = f"the {o.lower()}"
    elif cat == "help": 
        o = random.choice(pronouns)
        # FIX: Accusative pronouns for objects
        o_eng = object_pronoun(o)
    elif cat == "mental": o = random.choice(things + places); o_eng = f"the {o.lower()}"
    else: o = random.choice(buyable); o_eng = obj_article(o)

    v_eng = conjugate(v, "NOW", is_first, is_plural)
    return f"{s} {v} {o}", f"{s_eng.capitalize()} {v_eng} {o_eng}."

def gen_negation():
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)
    v = random.choice(["WANT", "NEED", "LIKE", "HAVE", "KNOW", "UNDERSTAND", "EAT", "SEE"])
    
    if v == "EAT":
        o = random.choice(consumables)
    else:
        o = random.choice(things + tech + places + consumables)
    
    do_aux = "do" if (is_first or is_plural) else "does"
    v_base = "have" if v == "HAVE" else v.lower()
    return f"{s} NOT {v} {o}", f"{s_eng.capitalize()} {do_aux} not {v_base} {obj_article(o)}."

def gen_compound():
    s = random.choice(pronouns)
    s_eng, is_first, is_plural = subject_english(s)
    place = random.choice(places)
    v2 = random.choice(["BUY", "SEE", "MEET", "FIND", "GET"])
    
    if v2 == "BUY" or v2 == "GET": o = random.choice(buyable)
    elif v2 == "MEET": o = random.choice(people)
    else: o = random.choice(things + tech)
    
    t = random.choice(["YESTERDAY", "TODAY", "TOMORROW", "NOW"])
    v1_eng = conjugate("GO", t, is_first, is_plural)
    
    if t == "YESTERDAY":
        v2_eng = conjugate(v2, "YESTERDAY", is_first, is_plural)
        eng = f"{format_time(t)}, {s_eng} {v1_eng} to the {place.lower()} and {v2_eng} {obj_article(o)}."
    elif t == "TOMORROW":
        eng = f"{format_time(t)}, {s_eng} {v1_eng} to the {place.lower()} and {v2.lower()} {obj_article(o)}."
    else:
        eng = f"{format_time(t)}, {s_eng} {v1_eng} to the {place.lower()} to {v2.lower()} {obj_article(o)}."
        
    return f"{t} {s} GO {place} {v2} {o}", eng

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

def gen_adjective():
    t = random.choice(times)
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)
    is_person = random.choice([True, False])
    if is_person: o = random.choice(people); adj = random.choice(adj_people)
    else: o = random.choice(places + things + vehicles); adj = random.choice(adj_size + adj_quality)
    v_eng = conjugate("SEE", t, is_first, is_plural)
    return f"{t} {s} SEE {adj} {o}", f"{format_time(t)}, {s_eng} {v_eng} the {adj.lower()} {o.lower()}."

def gen_number():
    numbers = ["ZERO","ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN"]
    t = random.choice(times)
    s = random.choice(pronouns + people)
    s_eng, is_first, is_plural = subject_english(s)
    num = random.choice(numbers)
    o = random.choice(things + tech + vehicles) 
    pluralized = o.lower() if num in ["ONE","ZERO"] else p.plural(o.lower())
    if t == "YESTERDAY": v_eng = "had"
    elif t == "TOMORROW": v_eng = "will have"
    else: v_eng = "have" if (is_first or is_plural) else "has"
    return f"{t} {s} HAVE {num} {o}", f"{format_time(t)}, {s_eng} {v_eng} {num.lower()} {pluralized}."

def gen_sov():
    s = random.choice(pronouns)
    s_eng, is_first, is_plural = subject_english(s)
    cat = random.choice(["visit", "tech_general", "study", "buy_sell"])
    v = random.choice(verbs_dict[cat])
    
    if cat == "visit": 
        o = random.choice(places)
        # FIX: Correct prepositions for Arrive vs Leave vs Motion verbs
        if v == "ARRIVE": o_eng = f"at the {o.lower()}"
        elif v in ["GO","WALK","RUN","COME","MOVE","CLIMB","RETURN"]: o_eng = f"to the {o.lower()}"
        else: o_eng = f"the {o.lower()}"
    elif cat == "tech_general": o = random.choice(tech + things); o_eng = f"the {o.lower()}"
    elif cat == "study": o = random.choice(things + tech); o_eng = f"the {o.lower()}"
    else: o = random.choice(buyable); o_eng = obj_article(o)
    
    t = random.choice(times) if random.random() < 0.5 else "NOW"
    v_eng = conjugate(v, t, is_first, is_plural)
    
    if t == "NOW":
        english = f"{s_eng.capitalize()} {v_eng} {o_eng}."
    else:
        subj = "I" if s_eng == "I" else s_eng.lower()
        english = f"{format_time(t)}, {subj} {v_eng} {o_eng}."

    if random.random() < 0.5:
        return (f"{t} {o} {s} {v}" if t != "NOW" else f"{o} {s} {v}"), english 
    else:
        return (f"{t} {o} {v} {s}" if t != "NOW" else f"{o} {v} {s}"), english 

def gen_wh_question():
    template = random.choice(["what_name", "where_place", "who_person", "what_want", "why_feel", "how_many"])

    if template == "what_name":
        s = random.choice(pronouns)
        possessive = {"I": "your", "YOU": "your", "HE": "his", "SHE": "her", "WE": "our", "THEY": "their"}
        return f"NAME WHAT {s}", f"What is {possessive[s]} name?"
    elif template == "where_place":
        o = random.choice(places + things)
        art = "my" if random.random() < 0.5 else "the"
        gloss_art = "MY" if art == "my" else "THE"
        return f"{gloss_art} {o} WHERE", f"Where is {art} {o.lower()}?"
    elif template == "who_person":
        role = random.choice(["TEACHER", "DOCTOR", "BOSS", "FRIEND"])
        return f"WHO {role}", f"Who is the {role.lower()}?"
    elif template == "what_want":
        s = random.choice(pronouns)
        s_eng, is_first, is_plural = subject_english(s)
        do = "do" if (is_first or is_plural) else "does"
        return f"{s} WANT WHAT", f"What {do} {s_eng} want?"
    elif template == "why_feel":
        s = random.choice(pronouns)
        s_eng, is_first, is_plural = subject_english(s)
        f_ = random.choice(feelings)
        be = "am" if is_first else "are" if is_plural else "is"
        return f"WHY {s} FEEL {f_}", f"Why {be} {s_eng} feeling {f_.lower()}?"
    else:
        o = random.choice(things + tech + mass_nouns)
        s = random.choice(pronouns)
        s_eng, is_first, is_plural = subject_english(s)
        do = "do" if (is_first or is_plural) else "does"
        if o in mass_nouns:
            return f"HOW-MUCH {o} {s} HAVE", f"How much {o.lower()} {do} {s_eng} have?"
        return f"HOW-MANY {o} {s} HAVE", f"How many {p.plural(o.lower())} {do} {s_eng} have?"

def gen_yn_question():
    template = random.choice(["feel_q", "have_q", "go_q", "done_q"])
    s = random.choice(pronouns)
    s_eng, is_first, is_plural = subject_english(s)
    be = "am" if is_first else "are" if is_plural else "is"
    do = "do" if (is_first or is_plural) else "does"

    if template == "feel_q":
        f_ = random.choice(feelings)
        return f"{s} FEEL {f_} YOU-KNOW", f"{be.capitalize()} {s_eng} feeling {f_.lower()}?"
    elif template == "have_q":
        o = random.choice(things + tech)
        return f"{s} HAVE {o} YOU-KNOW", f"{do.capitalize()} {s_eng} have the {o.lower()}?"
    elif template == "go_q":
        place = random.choice(places)
        return f"{s} GO {place} YOU-KNOW", f"{be.capitalize()} {s_eng} going to the {place.lower()}?"
    else:
        v = random.choice(["EAT", "READ", "WRITE", "BUY", "FINISH"])
        past = {"EAT": "eaten", "READ": "read", "WRITE": "written", "BUY": "bought", "FINISH": "finished"}
        have_aux = "Have" if (is_first or is_plural) else "Has"
        return f"{s} FINISH {v} YOU-KNOW", f"{have_aux} {s_eng} {past.get(v, v.lower())} yet?"

def gen_imperative():
    template = random.choice([
        "please_verb", "please_verb", "please_help", 
        "stop_verb", "come_place", "please_obj"
    ])

    if template == "please_verb":
        v_map = {"WAIT": "wait", "LISTEN": "listen", "STOP": "stop", "SIT": "sit", "COME": "come", "GO": "go", "READ": "read", "WRITE": "write", "CALL": "call", "LOOK": "look", "HELP": "help"}
        v = random.choice(list(v_map.keys()))
        return f"PLEASE {v}", f"Please {v_map[v]}."
    elif template == "please_help":
        return "HELP ME PLEASE", "Please help me."
    elif template == "stop_verb":
        stop_map = {"STOP TALK": "talking", "STOP RUN": "running", "STOP EAT": "eating", "STOP WAIT": "waiting"}
        v = random.choice(list(stop_map.keys()))
        return v, f"Stop {stop_map[v]}."
    elif template == "come_place":
        place = random.choice(places)
        return f"COME {place}", f"Come to the {place.lower()}."
    else:
        v_obj_map = {"BRING": "Bring", "SHOW": "Show me", "GIVE": "Give me"}
        v_key = random.choice(list(v_obj_map.keys()))
        o = random.choice(things + tech)
        return f"PLEASE {v_key} {o}", f"{v_obj_map[v_key]} the {o.lower()}, please."

def gen_mental():
    s = random.choice(pronouns)
    s_eng, is_first, is_plural = subject_english(s)
    v = random.choice(verbs_dict["mental"])
    o = random.choice(things + places + tech + consumables)
    art = "some" if o in consumables else "the"
    v_eng = conjugate(v, "NOW", is_first, is_plural)
    return f"{s} {v} {o}", f"{s_eng.capitalize()} {v_eng} {art} {o.lower()}."

def gen_topic_comment():
    o = random.choice(things + tech)
    s = random.choice(pronouns)
    s_eng, is_first, is_plural = subject_english(s)
    v = random.choice(["READ", "USE", "NEED", "WANT", "HAVE", "LIKE"])
    v_eng = conjugate(v, "NOW", is_first, is_plural)
    return f"{o} {s} {v}", f"As for the {o.lower()}, {s_eng} {v_eng} it."

# =====================================================================
# 5. SAFETY NET & IDIOMS
# =====================================================================
def build_safety_net(word_list):
    noun_classes = set(places + things + tech + vehicles + consumables + people)
    rows = []
    for word in word_list:
        w = word.upper()
        wl = word.lower()
        if w not in noun_classes: continue
        rows.append({"gloss": f"MY {w} WHERE",  "text": f"Where is my {wl}?"})
        rows.append({"gloss": f"I NEED {w}",    "text": f"I need the {wl}."})
        rows.append({"gloss": f"YOU HAVE {w} YOU-KNOW",  "text": f"Do you have the {wl}?"}) 
        rows.append({"gloss": f"I LIKE {w}",    "text": f"I like the {wl}."})
    return rows

asl_idioms = [
    ("HELLO HOW YOU", "Hello, how are you?"),
    ("WHAT YOUR NAME", "What is your name?"),
    ("I LOVE YOU", "I love you."),
    ("THANKYOU GOODBYE", "Thank you, goodbye."),
    ("I NEED HELP", "I need help."),
    ("SEE YOU TOMORROW", "See you tomorrow."),
    ("HOW MANY MONEY", "How much money?"),
    ("I HAPPY", "I am happy."),
    ("I SAD", "I am sad."),
    ("YOU OKAY YOU", "Are you okay?"),
    ("WHERE BATHROOM", "Where is the bathroom?"),
    ("PLEASE HELP ME", "Please help me."),
    ("NICE MEET YOU", "Nice to meet you."),
    ("I NOT KNOW", "I do not know."),
    ("I NOT UNDERSTAND", "I do not understand."),
    ("PLEASE WAIT", "Please wait."),
    ("SORRY MY BAD", "Sorry, my bad."),
    ("YOU WANT EAT YOU-KNOW", "Do you want to eat?"),
    ("I WANT TO EAT FOOD", "I want to eat food.")
]

def generate_idiom_variations():
    idioms = []
    for _ in range(50):
        for gloss, text in asl_idioms:
            idioms.append({"gloss": gloss, "text": text})
            
    for _ in range(500):
        obj = random.choice(places + tech + things + vehicles + consumables)
        if obj in mass_nouns:
            idioms.append({"gloss": f"HOW MUCH {obj}", "text": f"How much {obj.lower()}?"})
        else:
            idioms.append({"gloss": f"HOW MANY {obj}", "text": f"How many {p.plural(obj.lower())}?"})
    return idioms

# =====================================================================
# 6. EXECUTE & EXPORT
# =====================================================================
generators = [
    (gen_time_svo,       20), 
    (gen_negation,       10), 
    (gen_compound,       10), 
    (gen_feeling,         5),
    (gen_adjective,       5), 
    (gen_number,          4),
    (gen_no_time_svo,    12),  
    (gen_sov,             8),
    (gen_wh_question,     8),
    (gen_yn_question,     5),
    (gen_imperative,     10),
    (gen_mental,          5), 
    (gen_topic_comment,   5),
]

pool = []
for fn, weight in generators:
    pool.extend([fn] * weight)

TARGET = 50000
data = []
imperative_data = []  

imperative_data.extend(generate_idiom_variations())

for _ in range(TARGET):
    fn = random.choice(pool)
    try:
        g, e = fn()
        if fn == gen_imperative:
            imperative_data.append({"gloss": g, "text": e})
        else:
            data.append({"gloss": g, "text": e})
    except Exception:
        pass

df_main = pd.DataFrame(data).sample(frac=1, random_state=42).drop_duplicates(subset=["gloss"])

df_imp = pd.DataFrame(imperative_data)
if len(df_imp):
    df_imp = df_imp.groupby("gloss").head(50).reset_index(drop=True)

safety = build_safety_net(all_words)
df_base = (pd.concat([df_main, pd.DataFrame(safety)], ignore_index=True)
             .drop_duplicates(subset=["gloss"])
             .reset_index(drop=True))

df_final = (pd.concat([df_base, df_imp], ignore_index=True)
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))

save_path = "slt_stage3_dataset_final.csv"
df_final.to_csv(save_path, index=False)

print(f"📊 Generated {TARGET} raw")
print(f"   Main template unique: {len(df_main)}")
print(f"   Imperative/Idiom rows (capped 50/gloss): {len(df_imp)}")
print(f"✅ Saved {len(df_final)} total pairs to {save_path}")