#!/usr/bin/env python3
"""Rule-generated ASL-order gloss sequences over the locked 100-sign vocabulary.

The deployed Stage-3 renderer was trained on 15,843 synthetic rows that are almost
entirely order-preserving: only 95 of 11,840 matchable rows reorder a gloss. It
therefore learned to insert function words between glosses in the order it received
them, which is wrong for every ASL topic-comment, object-fronted, time-fronted or
wh-final utterance.

This module generates the supervision that was missing. It emits gloss sequences in
genuine ASL order, each carrying the English-relevant structure that produced it, plus
optional recognizer-noise variants whose English target must omit the spurious gloss.

The locked vocabulary has no copula, no articles, no tense morphology and no NOT.
Negation is NO, tense comes from a time gloss, and the copula/articles must be supplied
by the renderer. Those are exactly the transforms the generated corpus has to teach.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
VOCABULARY_MANIFEST = ROOT / "active/v17/citizen100_manifest.json"


def locked_vocabulary(manifest: Path = VOCABULARY_MANIFEST) -> tuple[str, ...]:
    """The 100 canonical labels the recognizer can emit, in class-index order."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = sorted(payload["classes"], key=lambda row: row["class_index"])
    return tuple(str(row["canonical_label"]).upper() for row in rows)


# ---------------------------------------------------------------------------
# Semantic classes
#
# Sequences are built from frames rather than free combination so the generated
# glosses stay interpretable. A nonsense sequence teaches the renderer nothing and
# makes the English target unreliable.
# ---------------------------------------------------------------------------

SIGNERS = ("I", "YOU", "WE", "THEY", "HE")
POSSESSIVES = {"I": "MY", "YOU": "YOUR", "WE": "OUR"}

PEOPLE = ("CHILD", "DOCTOR", "FATHER", "FRIEND", "MAN", "MOTHER", "WOMAN", "FAMILY")
# Only these read naturally after MY/YOUR/OUR; "our woman" does not.
POSSESSABLE = ("CHILD", "DOCTOR", "FATHER", "FRIEND", "MOTHER", "FAMILY")
PLACES = ("HOME", "SCHOOL", "HOSPITAL")
THINGS = ("WATER", "NAME", "LANGUAGE", "SIGN", "TIME", "DAY", "WEEK", "YEAR")

TIME_GLOSSES = ("NOW", "TOMORROW", "YESTERDAY", "MORNING", "NIGHT")
# Tense the renderer must recover, since no gloss carries it.
TIME_TENSE = {
    "NOW": "present",
    "TOMORROW": "future",
    "YESTERDAY": "past",
    "MORNING": "habitual",
    "NIGHT": "habitual",
}

PERSON_STATES = (
    "ANGRY", "BAD", "EXCITED", "GOOD", "HAPPY", "HUNGRY",
    "READY", "SAD", "SICK", "TIRED", "COLD", "HOT",
)
THING_STATES = ("BIG", "SMALL", "COLD", "HOT", "GOOD", "BAD", "EASY", "IMPORTANT")

INTRANSITIVE = ("SLEEP", "EAT", "WORK", "READ", "WRITE", "TALK", "LISTEN", "WAIT", "COME")
MOTION = ("GO", "COME")
TRANSITIVE_THING = ("WANT", "NEED", "HAVE", "LIKE", "LOVE", "SEE", "FIND",
                    "KNOW", "UNDERSTAND", "USE", "TAKE", "MAKE", "READ", "WRITE")
TRANSITIVE_PERSON = ("HELP", "SEE", "LOVE", "LIKE", "ASK", "TELL", "FIND", "KNOW", "HEAR")
COMPLEMENT_VERBS = ("WANT", "NEED", "LIKE", "TRY", "LEARN")
COMPLEMENT_ACTS = ("EAT", "SLEEP", "WORK", "READ", "WRITE", "GO", "LEARN", "TALK", "DRINK")

FIXED_PHRASES: tuple[tuple[str, ...], ...] = (
    ("HELLO",), ("GOODBYE",), ("THANKYOU",), ("SORRY",), ("YES",), ("NO",),
    ("PLEASE", "HELP"), ("PLEASE", "WAIT"), ("HELLO", "GOOD", "MORNING"),
    ("GOOD", "MORNING"), ("GOOD", "NIGHT"), ("THANKYOU", "GOODBYE"),
)


@dataclass(frozen=True)
class Utterance:
    """One generated example.

    ``glosses`` is what the recognizer would emit. ``meaning`` is a structured
    description handed to the English generator so it renders the intended reading
    rather than guessing from gloss order. ``noise_indices`` marks positions the
    English target must omit.
    """

    glosses: tuple[str, ...]
    structure: str
    meaning: str
    confidences: tuple[float, ...] = ()
    noise_indices: tuple[int, ...] = ()
    # True when the omitted gloss was recognized confidently and must still be left
    # out. A stranded determiner is the clear case: the recognizer really saw MY, but
    # with no noun after it there is nothing for the renderer to possess, and a model
    # that tries to honour it invents one.
    noise_is_confident: bool = False

    @property
    def key(self) -> str:
        return " ".join(self.glosses)

    @property
    def clean_glosses(self) -> tuple[str, ...]:
        return tuple(g for i, g in enumerate(self.glosses) if i not in set(self.noise_indices))

    def with_confidences(self, values: Sequence[float]) -> "Utterance":
        if len(values) != len(self.glosses):
            raise ValueError("one confidence per gloss is required")
        return Utterance(
            glosses=self.glosses,
            structure=self.structure,
            meaning=self.meaning,
            confidences=tuple(round(float(v), 3) for v in values),
            noise_indices=self.noise_indices,
            noise_is_confident=self.noise_is_confident,
        )


def _person_phrase(rng: random.Random) -> tuple[tuple[str, ...], str]:
    """A subject noun phrase and its English description."""
    choice = rng.random()
    if choice < 0.55:
        signer = rng.choice(SIGNERS)
        return (signer,), signer.lower()
    if choice < 0.85:
        owner = rng.choice(("MY", "YOUR", "OUR"))
        person = rng.choice(POSSESSABLE)
        return (owner, person), f"{owner.lower()} {person.lower()}"
    person = rng.choice(PEOPLE)
    return (person,), f"the {person.lower()}"


# ---------------------------------------------------------------------------
# Structure builders
#
# Each returns an Utterance in ASL order. The ``meaning`` string states the English
# reading explicitly, because gloss order alone is what the current model misreads.
# ---------------------------------------------------------------------------

def build_state(rng: random.Random) -> Utterance:
    """Adjectival predicate with no copula: I TIRED / MY MOTHER SICK.

    YOU is excluded deliberately. A bare "YOU TIRED" carries question marking on the
    face, which the gloss stream does not record, and in conversation it is nearly
    always a question. Letting it appear here as a statement too would give the same
    gloss sequence two contradictory targets.
    """
    subject, subject_en = _person_phrase(rng)
    while subject == ("YOU",):
        subject, subject_en = _person_phrase(rng)
    state = rng.choice(PERSON_STATES)
    return Utterance(
        glosses=(*subject, state),
        structure="state_predicate",
        meaning=f"{subject_en} is/are {state.lower()} (supply the missing copula)",
    )


def build_state_list(rng: random.Random) -> Utterance:
    """Several states at once, which is how people actually report how they feel."""
    subject, subject_en = _person_phrase(rng)
    while subject == ("YOU",):
        subject, subject_en = _person_phrase(rng)
    states = rng.sample(PERSON_STATES, rng.choice((2, 2, 3)))
    feel = rng.random() < 0.5
    glosses = (*subject, "FEEL", *states) if feel else (*subject, *states)
    listed = ", ".join(s.lower() for s in states[:-1]) + f" and {states[-1].lower()}"
    return Utterance(
        glosses=glosses,
        structure="state_list",
        meaning=(
            f"{subject_en} {'feels' if feel else 'is/are'} {listed}. "
            f"Coordinate the states into one list"
        ),
    )


# Only possessives. MORE, LESS and SAME are legitimate before a noun ("less time")
# and tolerable before an adjective, so treating them as artifacts would teach the
# renderer to delete real modifiers.
STRANDED = ("MY", "YOUR", "OUR")


def build_stranded(rng: random.Random) -> Utterance:
    """A determiner or modifier left with nothing to attach to.

    Observed live: `I SICK MY HUNGRY` at confidences 0.97/0.89/0.91/0.52 rendered as
    "I am sick, and my family is hungry." The recognizer saw MY clearly, but no noun
    followed, so the renderer supplied the commonest possessed noun and invented
    content the signer never produced. The gloss must be left out on grammatical
    grounds while its confidence stays high.
    """
    subject, subject_en = _person_phrase(rng)
    while subject == ("YOU",):
        subject, subject_en = _person_phrase(rng)
    stranded = rng.choice(STRANDED)
    # The determiner must land before something it cannot determine. Before a noun a
    # possessive is perfectly grammatical, so only the adjective case is an artifact.
    states = rng.sample(PERSON_STATES, 2)
    base = (*subject, states[0])
    tail = (states[1],)
    reading = f"{subject_en} is/are {states[0].lower()} and {states[1].lower()}"
    glosses = (*base, stranded, *tail)
    return Utterance(
        glosses=glosses,
        structure="stranded",
        meaning=(
            f"{reading}. The recognizer also emitted {stranded} at position "
            f"{len(base) + 1}. It was seen clearly but nothing follows it that it can "
            f"attach to, so it is a recognition artifact: leave it out entirely and do "
            f"NOT invent a noun for it"
        ),
        noise_indices=(len(base),),
        noise_is_confident=True,
    )


def build_stranded_final(rng: random.Random) -> Utterance:
    """The same artifact at the end of a buffer, where it has nothing at all to modify."""
    inner = rng.choice((build_state, build_svo, build_complement))(rng)
    stranded = rng.choice(STRANDED)
    return Utterance(
        glosses=(*inner.glosses, stranded),
        structure="stranded",
        meaning=(
            f"{inner.meaning}. The recognizer also emitted a trailing {stranded} with "
            f"nothing after it; leave it out entirely and do NOT invent a noun for it"
        ),
        noise_indices=(len(inner.glosses),),
        noise_is_confident=True,
    )


def build_greeting_plus(rng: random.Random) -> Utterance:
    """A greeting attached to what follows, so the greeting is not swallowed."""
    greeting = rng.choice(("HELLO", "GOODBYE", "THANKYOU"))
    inner = rng.choice((build_wh_final, build_wh_initial, build_state, build_yes_no))(rng)
    return Utterance(
        glosses=(greeting, *inner.glosses),
        structure=f"greeting+{inner.structure}",
        meaning=(
            f"opens with {greeting.lower()}, which must be kept. Then: {inner.meaning}"
        ),
    )


def build_svo(rng: random.Random) -> Utterance:
    """Plain subject-verb-object, the case the current model already handles."""
    subject, subject_en = _person_phrase(rng)
    if rng.random() < 0.5:
        verb = rng.choice(TRANSITIVE_THING)
        obj = rng.choice(THINGS)
    else:
        verb = rng.choice(TRANSITIVE_PERSON)
        obj = rng.choice([p for p in PEOPLE if p not in subject])
    return Utterance(
        glosses=(*subject, verb, obj),
        structure="svo",
        meaning=f"{subject_en} {verb.lower()} the {obj.lower()}",
    )


def build_osv(rng: random.Random) -> Utterance:
    """Object-fronted topic-comment: WATER I WANT means 'I want water'.

    This is the structure the deployed model gets wrong most visibly, because it
    renders the fronted object as the grammatical subject.
    """
    subject, subject_en = _person_phrase(rng)
    if rng.random() < 0.65:
        verb = rng.choice(TRANSITIVE_THING)
        obj = rng.choice(THINGS)
    else:
        verb = rng.choice(TRANSITIVE_PERSON)
        obj = rng.choice([p for p in PEOPLE if p not in subject])
    return Utterance(
        glosses=(obj, *subject, verb),
        structure="osv_topic",
        meaning=(
            f"topic-comment: the topic is {obj.lower()}; "
            f"{subject_en} {verb.lower()} it. The English subject is {subject_en}, "
            f"NOT {obj.lower()}"
        ),
    )


def build_motion(rng: random.Random) -> Utterance:
    """Movement toward a place, optionally with the place fronted as topic."""
    subject, subject_en = _person_phrase(rng)
    verb = rng.choice(MOTION)
    # A destination can be a person: GO DOCTOR means going to the doctor. Without
    # these rows the renderer only ever learns GO + building and rewrites the person
    # into the nearest place it knows.
    if rng.random() < 0.3:
        target = rng.choice([p for p in ("DOCTOR", "MOTHER", "FATHER", "FRIEND",
                                         "FAMILY") if p not in subject])
        return Utterance(
            glosses=(*subject, verb, target),
            structure="motion_person",
            meaning=f"{subject_en} {verb.lower()} to the {target.lower()}",
        )
    place = rng.choice(PLACES)
    if rng.random() < 0.35:
        return Utterance(
            glosses=(place, *subject, verb),
            structure="osv_place",
            meaning=(
                f"topic-comment: the topic is {place.lower()}; {subject_en} "
                f"{verb.lower()} there. The English subject is {subject_en}"
            ),
        )
    return Utterance(
        glosses=(*subject, verb, place),
        structure="motion",
        meaning=f"{subject_en} {verb.lower()} to {place.lower()}",
    )


def build_time_fronted(rng: random.Random) -> Utterance:
    """Time gloss first, which is where ASL puts it and where tense comes from."""
    time = rng.choice(TIME_GLOSSES)
    inner = rng.choice((build_state, build_svo, build_motion, build_complement))(rng)
    tense = TIME_TENSE[time]
    return Utterance(
        glosses=(time, *inner.glosses),
        structure=f"time_fronted+{inner.structure}",
        meaning=(
            f"time is fronted: {time.lower()} sets a {tense} reading. "
            f"Then: {inner.meaning}"
        ),
    )


def build_complement(rng: random.Random) -> Utterance:
    """Verb taking a verbal complement: I WANT EAT means 'I want to eat'."""
    subject, subject_en = _person_phrase(rng)
    verb = rng.choice(COMPLEMENT_VERBS)
    act = rng.choice([a for a in COMPLEMENT_ACTS if a != verb])
    return Utterance(
        glosses=(*subject, verb, act),
        structure="complement",
        meaning=f"{subject_en} {verb.lower()} to {act.lower()} (supply the infinitive 'to')",
    )


def build_wh_final(rng: random.Random) -> Utterance:
    """Wh-sign in final position, the ASL default: YOUR NAME WHAT."""
    kind = rng.choice(("name", "where", "when", "who", "why", "how"))
    if kind == "name":
        owner = rng.choice(("MY", "YOUR", "OUR"))
        return Utterance(
            glosses=(owner, "NAME", "WHAT"),
            structure="wh_final",
            meaning=f"a question asking what {owner.lower()} name is",
        )
    subject, subject_en = _person_phrase(rng)
    if kind == "where":
        verb = rng.choice(MOTION)
        return Utterance(
            glosses=(*subject, verb, "WHERE"),
            structure="wh_final",
            meaning=f"a question asking where {subject_en} {verb.lower()}",
        )
    if kind == "when":
        act = rng.choice(COMPLEMENT_ACTS)
        return Utterance(
            glosses=(*subject, act, "WHEN"),
            structure="wh_final",
            meaning=f"a question asking when {subject_en} {act.lower()}",
        )
    if kind == "who":
        verb = rng.choice(TRANSITIVE_PERSON)
        return Utterance(
            glosses=(*subject, verb, "WHO"),
            structure="wh_final",
            meaning=f"a question asking who {subject_en} {verb.lower()}",
        )
    if kind == "why":
        state = rng.choice(PERSON_STATES)
        return Utterance(
            glosses=(*subject, state, "WHY"),
            structure="wh_final",
            meaning=f"a question asking why {subject_en} is/are {state.lower()}",
        )
    act = rng.choice(("FEEL", "WORK", "LEARN"))
    return Utterance(
        glosses=(*subject, act, "HOW"),
        structure="wh_final",
        meaning=f"a question asking how {subject_en} {act.lower()}",
    )


def build_wh_initial(rng: random.Random) -> Utterance:
    """Wh-sign fronted, which also occurs and must not be read as a statement."""
    wh = rng.choice(("WHAT", "WHERE", "WHEN", "WHO", "WHY", "HOW"))
    subject, subject_en = _person_phrase(rng)
    if wh == "WHAT":
        verb = rng.choice(("WANT", "NEED", "SEE", "EAT", "MAKE", "READ"))
        return Utterance(
            glosses=(wh, *subject, verb),
            structure="wh_initial",
            meaning=f"a question asking what {subject_en} {verb.lower()}",
        )
    if wh == "WHERE":
        return Utterance(
            glosses=(wh, *subject, rng.choice(MOTION)),
            structure="wh_initial",
            meaning=f"a question asking where {subject_en} goes",
        )
    if wh == "WHO":
        return Utterance(
            glosses=(wh, *subject, rng.choice(TRANSITIVE_PERSON)),
            structure="wh_initial",
            meaning=f"a question asking who {subject_en} acts on",
        )
    if wh == "WHEN":
        return Utterance(
            glosses=(wh, *subject, rng.choice(COMPLEMENT_ACTS)),
            structure="wh_initial",
            meaning=f"a question asking when {subject_en} does that",
        )
    if wh == "WHY":
        return Utterance(
            glosses=(wh, *subject, rng.choice(PERSON_STATES)),
            structure="wh_initial",
            meaning=f"a question asking why {subject_en} feels that way",
        )
    return Utterance(
        glosses=(wh, *subject, rng.choice(("FEEL", "WORK"))),
        structure="wh_initial",
        meaning=f"a question asking how {subject_en} is doing",
    )


def build_wh_copula(rng: random.Random) -> Utterance:
    """A question that is a wh-sign plus a noun phrase, with no verb at all.

    These dominate the user's real sessions — WHO YOU, WHERE YOUR FAMILY,
    WHAT TIME TOMORROW, HOW YOUR DAY — and the locked vocabulary has no copula to
    carry them, so the renderer must supply "is"/"are". Without these rows it invents
    a verb instead and WHO YOU becomes "Who do you see?".
    """
    kind = rng.random()
    if kind < 0.3:
        who = rng.choice(("YOU", "HE", "THEY"))
        glosses = ("WHO", who) if rng.random() < 0.7 else (who, "WHO")
        return Utterance(
            glosses=glosses,
            structure="wh_copula",
            meaning=(
                f"a question asking who {who.lower()} is/are. There is no verb: supply "
                f"the copula and do not invent an action"
            ),
        )
    owner = rng.choice(("MY", "YOUR", "OUR"))
    thing = rng.choice(("FAMILY", "FRIEND", "MOTHER", "FATHER", "CHILD", "DOCTOR",
                        "NAME", "DAY", "WORK", "SCHOOL", "HOME"))
    wh = rng.choice(("WHERE", "HOW", "WHAT", "WHEN"))
    glosses = (wh, owner, thing) if rng.random() < 0.75 else (owner, thing, wh)
    return Utterance(
        glosses=glosses,
        structure="wh_copula",
        meaning=(
            f"a question asking {wh.lower()} {owner.lower()} {thing.lower()} is. There "
            f"is no verb: supply the copula and do not invent an action"
        ),
    )


def build_wh_time(rng: random.Random) -> Utterance:
    """WHAT plus a time or thing noun, another verbless real-session pattern."""
    noun = rng.choice(("TIME", "DAY", "WORK", "NAME", "YEAR", "WEEK"))
    extra = rng.choice((None, "TOMORROW", "NOW", "YESTERDAY", "MORNING", "NIGHT"))
    glosses = ("WHAT", noun) if extra is None else ("WHAT", noun, extra)
    when = "" if extra is None else f" {extra.lower()}"
    return Utterance(
        glosses=glosses,
        structure="wh_copula",
        meaning=(
            f"a question asking what the {noun.lower()} is{when}. There is no verb: "
            f"supply the copula and do not invent an action"
        ),
    )


def build_yes_no(rng: random.Random) -> Utterance:
    """A polar question, marked in ASL by non-manual signals the glosses do not carry.

    The renderer has to infer the question from the second-person subject and context;
    these rows teach that YOU-initial predicates are usually questions.
    """
    state_or_act = rng.random()
    if state_or_act < 0.4:
        state = rng.choice(PERSON_STATES)
        return Utterance(
            glosses=("YOU", state),
            structure="yes_no",
            meaning=f"a yes/no question asking whether you are {state.lower()}",
        )
    if state_or_act < 0.7:
        verb = rng.choice(COMPLEMENT_VERBS)
        act = rng.choice(COMPLEMENT_ACTS)
        return Utterance(
            glosses=("YOU", verb, act),
            structure="yes_no",
            meaning=f"a yes/no question asking whether you {verb.lower()} to {act.lower()}",
        )
    place = rng.choice(PLACES)
    return Utterance(
        glosses=("YOU", "GO", place),
        structure="yes_no",
        meaning=f"a yes/no question asking whether you are going to {place.lower()}",
    )


def build_negation(rng: random.Random) -> Utterance:
    """Negation with NO, since the locked vocabulary has no NOT."""
    subject, subject_en = _person_phrase(rng)
    if rng.random() < 0.5:
        verb = rng.choice(TRANSITIVE_THING)
        obj = rng.choice(THINGS)
        return Utterance(
            glosses=(*subject, "NO", verb, obj),
            structure="negation",
            meaning=(
                f"negated: {subject_en} do/does NOT {verb.lower()} the {obj.lower()}. "
                f"NO here is sentence negation, not the word 'no'"
            ),
        )
    state = rng.choice(PERSON_STATES)
    return Utterance(
        glosses=(*subject, "NO", state),
        structure="negation",
        meaning=(
            f"negated: {subject_en} is/are NOT {state.lower()}. "
            f"NO here is sentence negation, not the word 'no'"
        ),
    )


def build_possessive(rng: random.Random) -> Utterance:
    """Equative with no copula: MY FATHER DOCTOR."""
    owner = rng.choice(("MY", "YOUR", "OUR"))
    person = rng.choice(("FATHER", "MOTHER", "FRIEND", "CHILD"))
    role = rng.choice([r for r in ("DOCTOR", "MAN", "WOMAN", "FRIEND") if r != person])
    return Utterance(
        glosses=(owner, person, role),
        structure="equative",
        meaning=f"{owner.lower()} {person.lower()} is a {role.lower()} (supply the copula)",
    )


def build_conjoined(rng: random.Random) -> Utterance:
    """Two clauses run together, which is what the live buffer actually produces."""
    first = rng.choice((build_state, build_svo, build_negation))(rng)
    second = rng.choice((build_motion, build_complement, build_time_fronted))(rng)
    return Utterance(
        glosses=(*first.glosses, *second.glosses),
        structure=f"conjoined({first.structure}+{second.structure})",
        meaning=(
            f"two clauses in sequence. First: {first.meaning}. "
            f"Second: {second.meaning}. Join them naturally"
        ),
    )


def build_polite(rng: random.Random) -> Utterance:
    """Politeness markers, which attach rather than acting as predicates."""
    marker = rng.choice(("PLEASE", "THANKYOU", "SORRY"))
    if marker == "PLEASE":
        act = rng.choice(("HELP", "WAIT", "LISTEN", "COME", "STOP", "TELL"))
        glosses = (act, marker) if rng.random() < 0.5 else (marker, act)
        return Utterance(
            glosses=glosses,
            structure="polite",
            meaning=f"a polite request to {act.lower()}",
        )
    inner = build_state(rng)
    return Utterance(
        glosses=(marker, *inner.glosses),
        structure="polite",
        meaning=f"{marker.lower()} followed by: {inner.meaning}",
    )


def build_fixed(rng: random.Random) -> Utterance:
    """Short greetings and closings that appear constantly in live sessions."""
    glosses = rng.choice(FIXED_PHRASES)
    return Utterance(
        glosses=glosses,
        structure="fixed",
        meaning="a short conventional greeting, closing or response",
    )


# Weights reflect what the renderer has to get right, not natural frequency.
# The reordering structures are deliberately over-represented because they are the
# failure being repaired; plain SVO is kept present as a regression guard.
STRUCTURE_WEIGHTS: tuple[tuple[str, object, float], ...] = (
    ("osv_topic", build_osv, 0.11),
    ("time_fronted", build_time_fronted, 0.11),
    ("wh_final", build_wh_final, 0.11),
    ("state_predicate", build_state, 0.08),
    ("svo", build_svo, 0.08),
    ("complement", build_complement, 0.07),
    ("negation", build_negation, 0.06),
    ("motion", build_motion, 0.07),
    ("wh_initial", build_wh_initial, 0.05),
    ("wh_copula", build_wh_copula, 0.06),
    ("wh_time", build_wh_time, 0.03),
    ("yes_no", build_yes_no, 0.06),
    ("conjoined", build_conjoined, 0.04),
    ("state_list", build_state_list, 0.05),
    ("stranded", build_stranded, 0.04),
    ("stranded_final", build_stranded_final, 0.02),
    ("greeting_plus", build_greeting_plus, 0.04),
    ("equative", build_possessive, 0.02),
    ("polite", build_polite, 0.02),
    ("fixed", build_fixed, 0.01),
)


def generate_clean(count: int, seed: int) -> list[Utterance]:
    """Deduplicated clean utterances drawn from the weighted structure mix."""
    rng = random.Random(seed)
    builders = [builder for _, builder, _ in STRUCTURE_WEIGHTS]
    weights = [weight for _, _, weight in STRUCTURE_WEIGHTS]
    seen: set[str] = set()
    out: list[Utterance] = []
    # Bounded so an exhausted structure mix cannot spin forever.
    for _ in range(count * 60):
        if len(out) >= count:
            break
        utterance = rng.choices(builders, weights=weights, k=1)[0](rng)
        if utterance.key in seen:
            continue
        seen.add(utterance.key)
        out.append(utterance)
    return out


# ---------------------------------------------------------------------------
# Recognizer noise
#
# Confidence statistics are measured from the user's own 21 app sessions
# (artifacts/app_sessions/*/history.json): 2,006 accepted predictions with median
# score 0.615, p10 0.315 and minimum 0.250, against 811 rejected with median 0.189.
# Noise glosses are therefore injected in the 0.25-0.36 band, where LESS and YEAR
# actually slipped through in those sessions, while genuine glosses are drawn from
# the healthy part of the accepted distribution.
# ---------------------------------------------------------------------------

ACCEPTED_MEDIAN = 0.615
ACCEPTED_P10 = 0.315
ACCEPTED_MIN = 0.250

# The two bands overlap deliberately. Genuine signs really are accepted as low as
# 0.250 in the live sessions, so a disjoint split would teach "low score means drop"
# as a perfect rule and the renderer would delete real content whenever the
# recognizer was merely unsure. Confidence has to be informative, not decisive:
# the renderer must still weigh whether the gloss makes sense where it sits.
GENUINE_BAND = (0.250, 1.000, 0.700)   # low, high, mode
NOISE_BAND = (0.250, 0.750, 0.300)

# Signs the recognizer inserted spuriously in real sessions, plus short signs that
# are plausible transition artifacts. Sampling from a fixed pool keeps the noise
# supervision honest rather than letting any gloss appear anywhere.
NOISE_POOL = (
    "LESS", "YEAR", "TIME", "MORE", "SAME", "DAY", "WEEK", "SIGN",
    "MAYBE", "HAVE", "TAKE", "MAKE", "STOP", "SEE", "GIVE", "TRY",
)


def assign_confidences(utterance: Utterance, rng: random.Random) -> Utterance:
    """Give every gloss a plausible recognizer score.

    Genuine glosses sit in the healthy accepted band; injected noise sits in the weak
    band where real false positives occurred. This is the signal the renderer needs in
    order to drop a gloss on evidence rather than on a guess about plausibility.
    """
    noise = set(utterance.noise_indices)
    values = []
    for index in range(len(utterance.glosses)):
        if index in noise and not utterance.noise_is_confident:
            values.append(rng.triangular(*NOISE_BAND))
        else:
            # Confidently-recognized but stranded glosses draw from the genuine band,
            # so the renderer cannot learn "drop" as a pure function of the score.
            values.append(rng.triangular(*GENUINE_BAND))
    return utterance.with_confidences(values)


def inject_noise(utterance: Utterance, rng: random.Random) -> Utterance:
    """Insert one spurious gloss the English target must ignore.

    Two shapes are produced: a foreign gloss from the noise pool, and an adjacent
    duplicate of a real gloss, which is what a held sign looks like after CTC collapse
    fails to merge it. Insertion never lands before the first gloss of a fixed phrase,
    because those are already handled by reviewed templates.
    """
    glosses = list(utterance.glosses)
    if not glosses:
        return utterance
    if rng.random() < 0.3:
        position = rng.randrange(len(glosses))
        spurious = glosses[position]
        insert_at = position + 1
        shape = "adjacent_duplicate"
    else:
        candidates = [g for g in NOISE_POOL if g not in glosses]
        if not candidates:
            return utterance
        spurious = rng.choice(candidates)
        insert_at = rng.randrange(1, len(glosses) + 1)
        shape = "spurious_gloss"
    glosses.insert(insert_at, spurious)
    return Utterance(
        glosses=tuple(glosses),
        structure=f"{utterance.structure}+noise({shape})",
        meaning=(
            f"{utterance.meaning}. The recognizer also emitted {spurious} at position "
            f"{insert_at + 1} with low confidence; it is a recognition error and must "
            f"NOT appear in the English"
        ),
        noise_indices=(insert_at,),
    )


def generate_corpus(
    count: int,
    seed: int,
    noise_rate: float = 0.35,
) -> list[Utterance]:
    """Clean and noisy utterances with confidences, deduplicated by gloss sequence.

    ``noise_rate`` is the share of rows carrying an injected error. It is higher than
    the live false-positive rate on purpose: a third of the corpus teaching omission
    is what gives the renderer a usable decision boundary, and the clean majority
    holds the line against dropping real content.
    """
    rng = random.Random(seed)
    clean = generate_clean(count, seed)
    seen: set[str] = set()
    out: list[Utterance] = []
    for utterance in clean:
        # Structures that already carry a deliberate artifact keep only that one;
        # inject_noise rewrites noise_indices and would discard the original.
        if not utterance.noise_indices and rng.random() < noise_rate:
            utterance = inject_noise(utterance, rng)
        if utterance.key in seen:
            continue
        seen.add(utterance.key)
        out.append(assign_confidences(utterance, rng))
    # Appended last so the random stream above is untouched.
    for utterance in single_gloss_rows():
        if utterance.key in seen:
            continue
        seen.add(utterance.key)
        out.append(assign_confidences(utterance, rng))
    return out


def single_gloss_rows() -> list[Utterance]:
    """One row per locked gloss on its own.

    The recognizer really does emit a lone sign — a saved session produced bare "I I"
    — and a renderer that has never seen a one-gloss buffer invents a predicate for it.
    The deployed checkpoint turns "I" into "Is that?" and "I SICK" into "Is I sick?";
    an untrained replacement instead turns "I" into "I am happy.", which is worse in
    kind because it states something the signer never signed.

    These are appended after the random draw so they cannot disturb the generator's
    stream, which keeps every previously generated sequence and its cached English
    valid.
    """
    return [
        Utterance(
            glosses=(gloss,),
            structure="single",
            meaning=(
                f"a single sign on its own: {gloss.lower()}. Render only that word in "
                f"its shortest natural English form, capitalised as a standalone "
                f"utterance and ending in a full stop. Do not invent a predicate, a "
                f"subject or a question for it"
            ),
        )
        for gloss in locked_vocabulary()
    ]


def corpus_signature(rows: Iterable[Utterance]) -> str:
    """Stable hash of the generated sequences, for run provenance."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row.key}\0{row.structure}\0{row.noise_indices}\n".encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# English validation
#
# The legacy Stage-3 corpus contains rows like "Please help me, I need the computer
# for the part" for glosses that never mentioned a computer. A generated corpus has
# the same failure mode, so every target is checked against the glosses it came from
# before it is allowed into training.
# ---------------------------------------------------------------------------

GLOSS_LEMMAS: dict[str, tuple[str, ...]] = {
    "ANGRY": ("angry", "anger"), "ANSWER": ("answer",), "ASK": ("ask",),
    "BAD": ("bad",), "BIG": ("big",), "CHILD": ("child", "children", "kid"),
    "COLD": ("cold",), "COME": ("come", "came", "coming"),
    "DAY": ("day", "today", "daily"), "DIFFERENT": ("differ",),
    "DOCTOR": ("doctor",), "DRINK": ("drink", "drank", "drinking"),
    "EASY": ("easy", "easi"), "EAT": ("eat", "ate", "eaten", "eating", "meal"),
    "EXCITED": ("excit",), "FAMILY": ("famil",), "FATHER": ("father", "dad"),
    "FEEL": ("feel", "felt", "feeling"), "FIND": ("find", "found", "finding"),
    "FRIEND": ("friend",), "GIVE": ("give", "gave", "given", "giving"),
    "GO": ("go", "goes", "went", "going", "gone"), "GOOD": ("good", "well"),
    "GOODBYE": ("goodbye", "bye", "farewell"), "HAPPY": ("happy", "happi"),
    "HAVE": ("have", "has", "had", "having"), "HE": ("he", "him", "his"),
    "HEAR": ("hear", "heard", "hearing"), "HELLO": ("hello", "hi"),
    "HELP": ("help",), "HOME": ("home",), "HOSPITAL": ("hospital",),
    "HOT": ("hot",), "HOW": ("how",), "HUNGRY": ("hungry", "hungr"),
    "I": ("i", "me", "my", "mine"), "IMPORTANT": ("important",),
    "KNOW": ("know", "knew", "known"), "LANGUAGE": ("language",),
    "LEARN": ("learn",), "LESS": ("less", "fewer"), "LIKE": ("like",),
    "LISTEN": ("listen",), "LOVE": ("love",),
    "MAKE": ("make", "made", "making"), "MAN": ("man", "men"),
    "MAYBE": ("maybe", "perhaps", "might"), "MORE": ("more",),
    "MORNING": ("morning",), "MOTHER": ("mother", "mom"),
    "MY": ("my", "mine"), "NAME": ("name",), "NEED": ("need",),
    "NIGHT": ("night", "tonight"),
    "NO": ("no", "not", "don", "doesn", "didn", "won", "isn", "aren", "wasn",
           "weren", "hasn", "haven", "hadn", "couldn", "wouldn", "shouldn",
           "cannot", "never", "nothing"),
    "NOW": ("now", "currently"), "OUR": ("our", "ours"), "PLEASE": ("please",),
    "READ": ("read", "reading"), "READY": ("ready",), "SAD": ("sad",),
    "SAME": ("same",), "SCHOOL": ("school",),
    "SEE": ("see", "saw", "seen", "seeing"), "SICK": ("sick", "ill"),
    "SIGN": ("sign",), "SLEEP": ("sleep", "slept", "sleeping"),
    "SMALL": ("small", "little"), "SORRY": ("sorry", "apolog"),
    "STOP": ("stop",), "TAKE": ("take", "took", "taken", "taking"),
    "TALK": ("talk", "speak", "spoke", "speaking"),
    "TELL": ("tell", "told", "telling"), "THANKYOU": ("thank", "you"),
    "THEY": ("they", "them", "their"), "THINK": ("think", "thought"),
    "TIME": ("time",), "TIRED": ("tired", "tir"), "TOMORROW": ("tomorrow",),
    "TRY": ("try", "tri", "attempt"),
    "UNDERSTAND": ("understand", "understood"),
    "USE": ("use", "using", "used"), "WAIT": ("wait",), "WANT": ("want",),
    "WATER": ("water",), "WE": ("we", "us", "our"), "WEEK": ("week",),
    "WHAT": ("what",), "WHEN": ("when",), "WHERE": ("where",),
    "WHO": ("who", "whom"), "WHY": ("why",), "WOMAN": ("woman", "women"),
    "WORK": ("work",), "WRITE": ("write", "wrote", "written", "writing"),
    "YEAR": ("year",), "YES": ("yes", "yeah"), "YESTERDAY": ("yesterday",),
    "YOU": ("you", "your", "yours"), "YOUR": ("your", "yours"),
}

# Closed-class words the renderer is expected to supply. These are exactly the
# insertions ASL glosses omit, so they can never count as invented content.
FUNCTION_WORDS = frozenset("""
a an the is are am was were be been being do does did done
to of in on at for with and or but that this these those it its there here
will would can could shall should may must
s t re ve ll d m
as by from up out about into if then so too very just still yet
him her them us me one
""".split())

# Glosses whose absence changes who the sentence is about. Unlike a fronted topic,
# these can never be left unrealized, so "I TIRED" may not become "Is it tired?".
OBLIGATORY_GLOSSES = frozenset({"I", "YOU", "WE", "THEY", "HE", "MY", "YOUR", "OUR"})


def _tokens(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalpha() else " " for c in text).split() if t]


# Suffixes an English renderer may add to a gloss stem. Restricting growth to these
# is what separates "USE" -> "uses" from the gloss I -> "is", which is how the
# deployed model's "I TIRED" -> "Is it tired?" was able to validate as correct.
_INFLECTIONS = ("", "s", "es", "ed", "d", "ing", "ies", "n", "en", "er", "est",
                "ly", "y", "ily", "ize", "ise", "izing", "ising")


def _covers(token: str, lemmas: Iterable[str]) -> bool:
    """Match a token against gloss lemmas, allowing only ordinary inflection.

    Lemmas shorter than three characters must match exactly; pronouns like I, HE and
    WE are too short for any prefix rule to be safe.
    """
    for lemma in lemmas:
        if token == lemma:
            return True
        if len(lemma) < 3:
            continue
        if token.startswith(lemma) and token[len(lemma):] in _INFLECTIONS:
            return True
        # Stems such as "happi" and "tir" are stored deliberately short.
        if len(token) >= 4 and lemma.startswith(token):
            return True
    return False


def validate_english(utterance: Utterance, english: str) -> tuple[bool, list[str]]:
    """Check a generated target against the glosses that produced it.

    Rejects invented content, a noise gloss that leaked into the sentence, and
    sentences that dropped more than one confident gloss. One unrealized gloss is
    tolerated because legitimate English pronominalizes a fronted topic: "MAN THEY
    FIND" is correctly "They found him".
    """
    reasons: list[str] = []
    text = english.strip()
    if not text:
        return False, ["empty"]
    if len(text) > 300:
        reasons.append("overlong")

    kept = utterance.clean_glosses
    dropped = tuple(utterance.glosses[i] for i in utterance.noise_indices)
    kept_lemmas = {lemma for gloss in kept for lemma in GLOSS_LEMMAS.get(gloss, ())}

    tokens = _tokens(text)
    unknown = [
        token for token in tokens
        if token not in FUNCTION_WORDS and not _covers(token, kept_lemmas)
    ]
    if unknown:
        reasons.append(f"invented_content:{','.join(sorted(set(unknown))[:5])}")

    # A dropped gloss has leaked only if nothing else in the sequence explains the word.
    for gloss in dropped:
        own = GLOSS_LEMMAS.get(gloss, ())
        shared = {lemma for g in kept for lemma in GLOSS_LEMMAS.get(g, ())}
        exclusive = [lemma for lemma in own if lemma not in shared]
        if exclusive and any(_covers(token, exclusive) for token in tokens):
            reasons.append(f"noise_leaked:{gloss}")

    missing = [
        gloss for gloss in kept
        if GLOSS_LEMMAS.get(gloss) and not any(
            _covers(token, GLOSS_LEMMAS[gloss]) for token in tokens
        )
    ]
    unrealized_subjects = [gloss for gloss in missing if gloss in OBLIGATORY_GLOSSES]
    if unrealized_subjects:
        reasons.append(f"lost_subject:{','.join(unrealized_subjects)}")
    # One unrealized content gloss is legitimate: English pronominalizes a fronted
    # topic, so "MAN THEY FIND" may correctly render as "They found him".
    if len([g for g in missing if g not in OBLIGATORY_GLOSSES]) > 1:
        reasons.append(f"dropped_content:{','.join(missing[:5])}")

    return not reasons, reasons
