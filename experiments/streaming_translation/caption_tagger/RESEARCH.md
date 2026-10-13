# Research notes: subtitle-style English from a gloss stream — 2026-10-13 PHT

Question from the user: English should read like live subtitles — mostly the
recognized word itself, plus a limited set of inserted words — instead of a
sentence that waits for Finish. Example: I GO HOSPITAL TOMORROW MORNING.

## 1. What the literature says

**Edit/tagging generation ("tag, don't rewrite").** When output mostly copies
input, predicting a small edit per input token beats free generation in
reliability, data efficiency and speed.
- LaserTagger (Malmi et al., EMNLP 2019): KEEP/DELETE plus "add phrase X before
  the token", X from a restricted phrase vocabulary; competitive with seq2seq,
  better with little data, >100× faster inference.
  https://aclanthology.org/D19-1510/
- GECToR (Omelianchuk et al., BEA 2020): KEEP/DELETE/APPEND_t/REPLACE_t plus
  grammatical "g-transformations" such as verb-form changes from a dictionary;
  ~10× faster than seq2seq. https://arxiv.org/abs/2005.12592
- EdiT5 (Mallinson et al., Findings EMNLP 2022): tag which tokens survive,
  reorder, then autoregressively insert only the missing tokens; up to 25× faster.
  https://arxiv.org/abs/2205.12209
- Levenshtein Transformer (Gu et al., NeurIPS 2019): insertion/deletion as the
  primitive operations. https://arxiv.org/abs/1905.11006

Application to ATLAS: gloss→English is that same situation. Each gloss becomes an
insertion before it from a closed inventory, a surface form (GECToR-style verb/
pronoun-form dictionary) and punctuation. Content words cannot be invented or
dropped by construction, which addresses the repeated Stage3 failures (dropped NO,
lost DOCTOR, swapped participants, I → "Is it here?").

**Streaming / simultaneous translation.**
- STACL wait-k (Ma et al., ACL 2019): prefix-to-prefix translation lagging k source
  words. Monotone gloss→English with insertions is the easy end of this problem.
  https://aclanthology.org/P19-1289/
- Re-translation vs streaming (Arivazhagan et al., IWSLT 2020): re-translation with
  normalized erasure NE < 0.2 is competitive; NE = 0 (append-only) is recommended
  as a baseline; prefix training is essential. https://arxiv.org/abs/2004.03643
  Phase 1 measured NE 0.91 (tiny T5) and 1.28 (Qwen): far into the unstable regime.
- Local agreement (Liu, Spanakis & Niehues, Interspeech 2020): commit the prefix
  that consecutive hypotheses agree on; used later by Whisper-Streaming.
  https://arxiv.org/abs/2005.11185 . Phase 1's sentence-agreement policy was a
  coarse version; it finalized only 3/20 phone tails.
- Incremental NLU (Madureira & Schlangen, EMNLP 2020): delaying output by 1–2
  tokens of right context, or training on truncated inputs, sharply reduces
  edit overhead (e.g. BERT POS EO 0.52 → 0.05 with truncated training).
  https://aclanthology.org/2020.emnlp-main.26 — metrics from Baumann, Buß &
  Schlangen (2011), https://doi.org/10.5087/dad.2011.106 . The causal tagger
  used here is exactly equivalent to prefix-trained, so prefix and full-input
  behavior match by construction.
- Sign-language streaming work (SimulSLT 2021, Online CSLR/SLT EMNLP 2024,
  adaptive length LREC-COLING 2024) was reviewed in phase 1; all use wait-k-like
  policies on PHOENIX/CSL weather/daily domains, not phone ASL.

**Gloss-to-text specifically.**
- Moryossef et al. (AT4SSL 2021) built synthetic ASL-like gloss from English by
  keeping only nouns, verbs, adjectives, adverbs and numerals, lemmatizing, and
  locally permuting (≤4 positions): +3.14 BLEU on NCSLGR ASL→English.
  https://aclanthology.org/2021.mtsummit-at4ssl.1 — i.e. the gloss is English
  minus function words and inflection; the inverse is a closed insertion problem.
- ASLG-PC12 (Othman & Jemni, LREC 2012): 100M+ English–gloss pairs produced by a
  rule-based English→gloss transform over POS tags.
  https://www.sign-lang.uni-hamburg.de/lrec/pub/12019.pdf
- Gloss2Text (Fayyazsanavi et al., Findings EMNLP 2024): LLM + semantic label
  smoothing on PHOENIX; full-sentence, not streaming.
  https://arxiv.org/abs/2407.01394

**ASL grammar facts that determine the insertion set.**
- No be-verb: HE MY BROTHER (Lifeprint). https://lifeprint.com/asl101/pages-layout/syntax.htm
- Common TIME–TOPIC–COMMENT order: time signs usually come first, so tense is
  typically known before the verb arrives (good for left-to-right captions).
  https://lifeprint.com/asl101/topics/topic-comment.htm
- Tense mostly from time adverbials rather than verb inflection (debated; Neidle
  et al. identify lexical tense markers). https://www.bu.edu/asllrp/rpt14/asllrpr14.pdf
- WH-signs are often clause-final (YOU LEARN SIGN WHERE), and negation can be a
  final NOT/NO or a headshake over the predicate.
  https://lifeprint.com/asl101/topics/sentencetypes.htm ;
  https://ling.upenn.edu/courses/Fall_1997/ling001/nmbk96.pdf
  These are the two cases where English word order truly differs from the stream.
- Null arguments are licensed in ASL (Bahan et al. 2000; Lillo-Martin 1986), so
  English may need an object/subject pronoun the signer did not produce
  ("please help [me]"). That is the one insertion type that can add meaning.

**Live-caption practice.**
- Google's live-caption stability study (Liu et al., CHI EA 2023, n=123):
  viewers "preferred stability over accuracy"; appending at the end is always
  applied, mid-caption changes only if they do not reflow lines, and re-captions
  only if meaning changes; stabilization significantly improved comfort/readability.
  https://research.google/blog/modeling-and-improving-text-stability-in-live-captions/
  (Crowdworkers, not specifically Deaf viewers.)
- DCMP Captioning Key: presentation-rate targets ~120–160 wpm by audience; two
  lines; editing allowed for readability. https://dcmp.org/learn/captioningkey/601
  ASL signing produces far fewer than 120 English words per minute, so rate is
  not the constraint; stability and faithfulness are.

## 2. What follows for ATLAS

1. A caption should be an *append-only edit stream*: render each gloss once its
   one-sign lookahead is known; never retranslate the whole buffer.
2. The model should be a small learned tagger (insertion class + form class +
   punctuation), not a free generator — consistent with the user's 2026-09-29
   rule against phrase triggers, and with the recorded semantic failures.
3. Two ASL constructions need either an echo-style caption ("You go where?") or
   one bounded sentence-level revision when the clause closes: clause-final WH
   and clause-final negation.
4. Inserted pronouns are the residual hallucination channel and must be measured.
5. Evaluation should report NE / edit overhead and gloss-order fidelity alongside
   BLEU/chrF, as phase 1 showed fluent text can be unfaithful.

Measured evidence for these points is in `REPORT.md`.
