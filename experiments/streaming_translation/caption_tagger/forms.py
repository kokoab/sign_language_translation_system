"""Surface-form dictionary and English tokenization for the 100-gloss vocabulary.

A fixed morphological inventory (like GECToR's verb-form table). Which form is
used in context is learned by the tagger, never chosen by rules here.
"""
import re

_RAW = {
    'ANGRY': 'angry|angrier|anger|angrily', 'ANSWER': 'answer|answers|answered|answering',
    'ASK': 'ask|asks|asked|asking', 'BAD': 'bad|badly|worse|worst',
    'BIG': 'big|bigger|biggest', 'CHILD': 'child|children|kid|kids',
    'COLD': 'cold|colder', 'COME': 'come|comes|came|coming',
    'DAY': 'day|days|daily|today', 'DIFFERENT': 'different|differently|difference',
    'DOCTOR': 'doctor|doctors', 'DRINK': 'drink|drinks|drank|drinking|drunk',
    'EASY': 'easy|easier|easily', 'EAT': 'eat|eats|ate|eating|eaten',
    'EXCITED': 'excited|exciting|excitement', 'FAMILY': 'family|families',
    'FATHER': 'father|fathers|dad', 'FEEL': 'feel|feels|felt|feeling|feelings',
    'FIND': 'find|finds|found|finding', 'FRIEND': 'friend|friends',
    'FS0': 'fs0', 'GIVE': 'give|gives|gave|giving|given',
    'GO': 'go|goes|went|going|gone', 'GOOD': 'good|well|better|best',
    'GOODBYE': 'goodbye|bye', 'HAPPY': 'happy|happier|happily|happiness',
    'HAVE': 'have|has|had|having', 'HE': 'he|him|his|himself|she|her',
    'HEAR': 'hear|hears|heard|hearing', 'HELLO': 'hello|hi',
    'HELP': 'help|helps|helped|helping', 'HOME': 'home|homes|house',
    'HOSPITAL': 'hospital|hospitals', 'HOT': 'hot|hotter',
    'HOW': 'how', 'HUNGRY': 'hungry', 'I': 'i|me|myself',
    'IMPORTANT': 'important|importance', 'KNOW': 'know|knows|knew|knowing|known',
    'LANGUAGE': 'language|languages', 'LEARN': 'learn|learns|learned|learning|learnt',
    'LESS': 'less|fewer|least', 'LIKE': 'like|likes|liked|liking',
    'LISTEN': 'listen|listens|listened|listening', 'LOVE': 'love|loves|loved|loving',
    'MAKE': 'make|makes|made|making', 'MAN': 'man|men',
    'MAYBE': 'maybe|perhaps|might|may', 'MORE': 'more|most',
    'MORNING': 'morning|mornings', 'MOTHER': 'mother|mothers|mom',
    'MY': 'my|mine', 'NAME': 'name|names|named',
    'NEED': 'need|needs|needed|needing', 'NIGHT': 'night|nights|tonight',
    'NO': 'no|not|never', 'NOW': 'now|today|currently',
    'OUR': 'our|ours', 'PLEASE': 'please', 'READ': 'read|reads|reading',
    'READY': 'ready', 'SAD': 'sad|sadder|sadness', 'SAME': 'same',
    'SCHOOL': 'school|schools', 'SEE': 'see|sees|saw|seeing|seen',
    'SICK': 'sick|ill', 'SIGN': 'sign|signs|signed|signing',
    'SLEEP': 'sleep|sleeps|slept|sleeping|asleep', 'SMALL': 'small|smaller|smallest|little',
    'SORRY': 'sorry', 'STOP': 'stop|stops|stopped|stopping',
    'TAKE': 'take|takes|took|taking|taken', 'TALK': 'talk|talks|talked|talking',
    'TELL': 'tell|tells|told|telling', 'THANKYOU': 'thank you|thanks|thank',
    'THEY': 'they|them|their|themselves', 'THINK': 'think|thinks|thought|thinking',
    'TIME': 'time|times', 'TIRED': 'tired',
    'TOMORROW': 'tomorrow', 'TRY': 'try|tries|tried|trying',
    'UNDERSTAND': 'understand|understands|understood|understanding',
    'USE': 'use|uses|used|using', 'WAIT': 'wait|waits|waited|waiting',
    'WANT': 'want|wants|wanted|wanting', 'WATER': 'water',
    'WE': 'we|us|ourselves', 'WEEK': 'week|weeks|weekly',
    'WHAT': 'what', 'WHEN': 'when', 'WHERE': 'where', 'WHO': 'who|whom',
    'WHY': 'why', 'WOMAN': 'woman|women', 'WORK': 'work|works|worked|working',
    'WRITE': 'write|writes|wrote|writing|written', 'YEAR': 'year|years',
    'YES': 'yes', 'YESTERDAY': 'yesterday|last', 'YOU': 'you|yourself',
    'YOUR': 'your|yours',
}
FORMS = {g: [tuple(f.split()) for f in v.split('|')] for g, v in _RAW.items()}
GLOSSES = sorted(FORMS)

_CONTRACT = [(r"\bcan't\b", 'can not'), (r"\bcannot\b", 'can not'), (r"\bwon't\b", 'will not'),
             (r"n't\b", ' not'), (r"'m\b", ' am'), (r"'re\b", ' are'), (r"'ll\b", ' will'),
             (r"'ve\b", ' have'), (r"'d\b", ' would')]

def tokenize(text):
    """Lowercase, expand contractions, split punctuation. Keeps possessive 's."""
    t = text.replace('’', "'").replace('FS0', 'fs0')
    t = t.lower()
    for a, b in _CONTRACT:
        t = re.sub(a, b, t)
    return re.findall(r"fs0|'s|[a-z]+|[.,!?;:]", t)

def split_counted(english):
    """'6: Hello, ... 4: We ...' -> list of sentence strings."""
    parts = re.split(r'(?:^|\s)\d+:\s', english.strip())
    return [p.strip() for p in parts if p.strip()]

def detokenize(tokens):
    """Caption text from tokens: capitalize sentence starts and 'I', attach punctuation."""
    out, cap = [], True
    for tok in tokens:
        if tok in '.,!?;:' or tok == "'s":
            if out and tok in '.,!?;:' and out[-1][-1:] in '.,!?;:':
                continue  # never render doubled punctuation ("FS0.." / ".,")
            if out: out[-1] += tok
            else: out.append(tok)
            if tok in '.!?': cap = True
            continue
        w = 'FS0' if tok == 'fs0' else ('I' if tok == 'i' else tok)
        if cap and w: w = w[0].upper() + w[1:]
        cap = False
        out.append(w)
    return ' '.join(out)
