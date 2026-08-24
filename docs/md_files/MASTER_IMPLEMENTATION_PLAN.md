# SLT System Master Implementation Plan

**Generated:** 2026-03-18
**Source:** Synthesized from Claude Expert Analysis, Gemini Review, and Accuracy Critique
**Goal:** Transform the SLT pipeline from a translator into a conversational system for capstone defense

---

## Executive Summary

After cross-referencing three independent analyses, the consensus is clear:

| Analysis | Primary Bottleneck | Agreed Priority |
|----------|-------------------|-----------------|
| Claude Expert | Stage 3 dataset quality | DATA FIRST |
| Gemini Review | Stage 3 dataset is "biggest bottleneck" | DATA FIRST |
| Accuracy Critique | Stage 2 domain gap (transitions) | TRAINING ALIGNMENT |

**Bottom Line:** Fix the data, then fix the training, then polish inference.

---

## Phase 1: Quick Wins (Day 1-2)

### 1.1 MediaPipe Confidence Alignment (30 minutes)

All analyses agree: train and inference use different MediaPipe configs.

**Current State:**
- `extract.py`: `min_detection_confidence=0.65`, `model_complexity=1`
- `camera_inference.py`: `MIN_DETECTION_CONF=0.80`, `MODEL_COMPLEXITY=0`

**Action:** Align to stricter thresholds (train with cleaner data):

```python
# extract.py - Update PipelineConfig
min_detection_conf: float = 0.80  # Was 0.65
min_tracking_conf: float = 0.80   # Was 0.65
model_complexity: int = 1         # Keep same

# camera_inference.py - Already correct, verify:
MIN_DETECTION_CONF = 0.80
MIN_TRACKING_CONF = 0.80
MODEL_COMPLEXITY = 0  # OK for faster inference
```

**Expected Impact:** 3-5% accuracy improvement
**Risk Level:** LOW - Simple config change

---

### 1.2 Label Smoothing Increase (5 minutes)

Critique confirms: with 240+ classes, 0.05 may be too low.

```python
# train_stage_1.py
label_smoothing = 0.10  # Was 0.05
```

**Expected Impact:** Marginal regularization benefit
**Risk Level:** LOW

---

## Phase 2: Stage 3 Dataset Overhaul (Day 2-4)

**All three analyses agree this is CRITICAL.**

### 2.1 Current Dataset Problems

| Issue | Current | Target |
|-------|---------|--------|
| Questions | 3.5% | 15-20% |
| Single-word glosses | 0% | 5-10% |
| Semantic validity | Many nonsense combos | Filter invalid |
| Paraphrase diversity | 1 output per gloss | 3-5 variations |
| Max sequence length | 6 tokens | 8-10 tokens |

### 2.2 Create Enhanced Dataset Generator

```python
# src/generate_stage3_data_v2.py

import pandas as pd
import random
from itertools import product

# ============= SINGLE-WORD GLOSSES =============
SINGLE_WORD_GLOSSES = [
    ("HELLO", ["Hello.", "Hi.", "Hey."]),
    ("GOODBYE", ["Goodbye.", "Bye.", "See you."]),
    ("THANK-YOU", ["Thank you.", "Thanks.", "Thanks a lot."]),
    ("PLEASE", ["Please.", "Please?", "If you please."]),
    ("YES", ["Yes.", "Yeah.", "Yep."]),
    ("NO", ["No.", "Nope.", "No way."]),
    ("SORRY", ["Sorry.", "I'm sorry.", "My apologies."]),
    ("HELP", ["Help!", "Help me!", "I need help."]),
    ("STOP", ["Stop.", "Stop!", "Please stop."]),
    ("WAIT", ["Wait.", "Wait!", "Hold on."]),
    ("OK", ["Okay.", "OK.", "Alright."]),
    ("WHAT", ["What?", "What is it?", "Pardon?"]),
    ("WHY", ["Why?", "Why is that?", "How come?"]),
    ("HOW", ["How?", "How so?", "In what way?"]),
    ("WHERE", ["Where?", "Where is it?", "Which place?"]),
    ("WHO", ["Who?", "Who is it?", "Which person?"]),
    ("WHEN", ["When?", "At what time?", "Which day?"]),
]

# ============= QUESTION TEMPLATES (Target: 15-20%) =============
QUESTION_TEMPLATES = [
    # Yes/No questions (use YOU-KNOW for ASL question marker)
    ("{SUBJECT} {VERB} YOU-KNOW", "Is {subject} {verb_present}?"),
    ("{SUBJECT} WANT {OBJECT} YOU-KNOW", "Does {subject} want {object}?"),
    ("{SUBJECT} LIKE {OBJECT} YOU-KNOW", "Does {subject} like {object}?"),
    ("{SUBJECT} HAVE {OBJECT} YOU-KNOW", "Does {subject} have {object}?"),
    ("{SUBJECT} NEED {OBJECT} YOU-KNOW", "Does {subject} need {object}?"),

    # WH-questions
    ("WHAT {SUBJECT} {VERB}", "What is {subject} {verb_present}?"),
    ("WHAT {SUBJECT} WANT", "What does {subject} want?"),
    ("WHERE {SUBJECT} GO", "Where is {subject} going?"),
    ("WHERE {SUBJECT} LIVE", "Where does {subject} live?"),
    ("WHO {VERB} {OBJECT}", "Who {verb_past} {object}?"),
    ("WHEN {SUBJECT} {VERB}", "When does {subject} {verb_base}?"),
    ("WHEN {SUBJECT} GO {PLACE}", "When is {subject} going to {place}?"),
    ("HOW {SUBJECT} FEEL", "How is {subject} feeling?"),
    ("HOW {SUBJECT} DO", "How is {subject} doing?"),
    ("WHY {SUBJECT} {VERB}", "Why does {subject} {verb_base}?"),
    ("WHY {SUBJECT} NOT {VERB}", "Why doesn't {subject} {verb_base}?"),

    # Complex questions
    ("{TIME} WHAT {SUBJECT} {VERB}", "{time}, what is {subject} {verb_present}?"),
    ("{TIME} WHERE {SUBJECT} GO", "{time}, where is {subject} going?"),
]

# ============= INVALID COMBINATIONS TO FILTER =============
INVALID_VERB_OBJECT_PAIRS = {
    'BUY': ['PASSWORD', 'NAME', 'IDEA', 'WORD', 'SENTENCE', 'LANGUAGE', 'WEATHER'],
    'SELL': ['PASSWORD', 'NAME', 'IDEA', 'WORD', 'SENTENCE', 'WEATHER'],
    'EAT': ['WATER', 'COFFEE', 'TEA', 'JUICE', 'MILK', 'CAR', 'PHONE'],
    'DRINK': ['FOOD', 'APPLE', 'BREAD', 'PIZZA', 'CAR', 'PHONE'],
    'DRIVE': ['RESTAURANT', 'SCHOOL', 'HOSPITAL', 'LIBRARY', 'FOOD', 'BOOK'],
    'COOK': ['CAR', 'PHONE', 'BOOK', 'WORD', 'NAME', 'PASSWORD'],
    'READ': ['FOOD', 'WATER', 'CAR', 'PHONE'],
}

def is_valid_combination(tokens):
    """Check if verb-object pairs are semantically valid."""
    for i, token in enumerate(tokens):
        if token in INVALID_VERB_OBJECT_PAIRS:
            for j in range(i + 1, len(tokens)):
                if tokens[j] in INVALID_VERB_OBJECT_PAIRS[token]:
                    return False
    return True

# ============= PARAPHRASE VARIATIONS =============
COMMON_PARAPHRASES = {
    "I GO STORE": [
        "I am going to the store.",
        "I'm going to the store.",
        "I'm heading to the store.",
    ],
    "HELLO HOW YOU": [
        "Hello, how are you?",
        "Hi, how are you doing?",
        "Hey, how's it going?",
    ],
    # Add more...
}

def generate_enhanced_dataset(original_csv_path, output_path):
    """Generate enhanced Stage 3 dataset."""
    rows = []

    # 1. Load and filter original data
    df = pd.read_csv(original_csv_path)
    for _, row in df.iterrows():
        tokens = row['gloss'].split()
        if is_valid_combination(tokens):
            rows.append({'gloss': row['gloss'], 'text': row['text']})

    print(f"After filtering invalid combos: {len(rows)} rows")

    # 2. Add single-word glosses (repeat for balance)
    for gloss, texts in SINGLE_WORD_GLOSSES:
        for text in texts:
            for _ in range(50):  # Repeat for balance
                rows.append({'gloss': gloss, 'text': text})

    # 3. Generate questions (target ~18% of dataset)
    # ... question generation logic ...

    # 4. Add paraphrases for common patterns
    for gloss, paraphrases in COMMON_PARAPHRASES.items():
        for text in paraphrases:
            rows.append({'gloss': gloss, 'text': text})

    # Save
    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_path, index=False)
    print(f"Total rows: {len(result_df)}")

    # Stats
    q_count = sum(1 for r in rows if '?' in r['text'])
    print(f"Questions: {q_count} ({100*q_count/len(rows):.1f}%)")
```

**Expected Impact:** Major improvement in translation naturalness
**Risk Level:** MEDIUM - Requires retraining Stage 3

---

## Phase 3: Stage 2 Training Fixes (Day 4-7)

### 3.1 Transition Frame Injection (CRITICAL)

**All analyses strongly agree on this.**

The critique's constraint: Must preserve `T % 32 == 0` or resample properly.

```python
# train_stage_2.py - Add to SyntheticCTCDataset

def _create_transition_frames(self, prev_clip, next_clip):
    """
    Create realistic transition between signs.

    CRITICAL: Kinematics must be recomputed from interpolated XYZ,
    not interpolated directly (per accuracy critique).
    """
    trans_len = random.randint(4, 12)

    # Get end/start XYZ positions only
    end_xyz = prev_clip[-1, :, :3]    # [42, 3]
    start_xyz = next_clip[0, :, :3]   # [42, 3]

    # Interpolate XYZ (with optional ease-in-out)
    use_ease = random.random() < 0.3
    t = np.linspace(0, 1, trans_len)
    if use_ease:
        t = t * t * (3 - 2 * t)  # Smoothstep

    alphas = t[:, None, None]
    trans_xyz = (1 - alphas) * end_xyz + alphas * start_xyz

    # RECOMPUTE kinematics from interpolated XYZ (not interpolate kinematics!)
    vel = np.zeros_like(trans_xyz)
    if trans_len > 2:
        vel[1:-1] = (trans_xyz[2:] - trans_xyz[:-2]) / 2.0
        vel[0] = vel[1]
        vel[-1] = vel[-2]

    acc = np.zeros_like(trans_xyz)
    if trans_len > 2:
        acc[1:-1] = (vel[2:] - vel[:-2]) / 2.0
        acc[0] = acc[1]
        acc[-1] = acc[-2]

    # Inherit mask
    prev_mask = prev_clip[-1, :, 9:10]
    next_mask = next_clip[0, :, 9:10]
    trans_mask = np.maximum(prev_mask, next_mask)
    trans_mask = np.tile(trans_mask, (trans_len, 1, 1))

    return np.concatenate([trans_xyz, vel, acc, trans_mask], axis=-1).astype(np.float32)

def __getitem__(self, idx):
    files, target_glosses = self.samples[idx]
    arrays = []
    valid_targets = []

    for i, (f, tgt) in enumerate(zip(files, target_glosses)):
        arr = np.load(self.data_path / f).astype(np.float32)
        if arr.shape != (32, 42, 10):
            continue

        # INJECT TRANSITION (35% probability)
        if arrays and random.random() < 0.35:
            transition = self._create_transition_frames(arrays[-1], arr)
            arrays.append(transition)

        arrays.append(arr)
        valid_targets.append(tgt)

    if len(arrays) == 0:
        return np.zeros((32, 42, 10), dtype=np.float32), []

    x = np.concatenate(arrays, axis=0)

    # IMPORTANT: Resample to maintain 32-frame alignment if needed
    # Or let the forward() handle variable lengths
    return x, valid_targets
```

**Expected Impact:** 15-25% WER reduction on continuous video
**Risk Level:** MEDIUM - Core training change, must test thoroughly

---

### 3.2 Temporal Speed Augmentation (HIGH)

**Critique's key constraint:** Must warp XYZ FIRST, then recompute kinematics.

```python
# train_stage_1.py and train_stage_2.py

def temporal_speed_warp(xyz, min_speed=0.75, max_speed=1.25):
    """
    Warp temporal axis to simulate fast/slow signing.

    Args:
        xyz: [T, 42, 3] raw XYZ positions (NOT the 10-channel tensor!)

    Returns:
        Warped XYZ [T, 42, 3]
    """
    T, V, C = xyz.shape
    speed = np.random.uniform(min_speed, max_speed)

    orig_t = np.linspace(0, 1, T)
    warp_amount = (speed - 1.0) * 0.5
    warped_t = orig_t + warp_amount * orig_t * (1 - orig_t) * 4
    warped_t = np.clip(warped_t, 0, 1)
    warped_t = np.sort(warped_t)

    flat = xyz.reshape(T, -1)
    warped_flat = np.zeros_like(flat)
    for c in range(flat.shape[1]):
        warped_flat[:, c] = np.interp(orig_t, warped_t, flat[:, c])

    return warped_flat.reshape(T, V, C).astype(np.float32)

def online_augment_with_speed(x, speed_prob=0.5, rotation_deg=10.0,
                               scale_lo=0.85, scale_hi=1.15, noise_std=0.003):
    """
    Augmentation that includes temporal speed warping.
    REPLACES existing online_augment().
    """
    B, T, N, C = x.shape
    device = x.device

    # Step 1: Speed warp on XYZ, then recompute kinematics
    if torch.rand(1).item() < speed_prob:
        xyz = x[..., :3].cpu().numpy()

        for b in range(B):
            xyz[b] = temporal_speed_warp(xyz[b])

        warped_xyz = torch.from_numpy(xyz).to(device)

        # Recompute velocity
        vel = torch.zeros_like(warped_xyz)
        vel[:, 1:-1] = (warped_xyz[:, 2:] - warped_xyz[:, :-2]) / 2.0
        vel[:, 0] = vel[:, 1]
        vel[:, -1] = vel[:, -2]

        # Recompute acceleration
        acc = torch.zeros_like(warped_xyz)
        acc[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / 2.0
        acc[:, 0] = acc[:, 1]
        acc[:, -1] = acc[:, -2]

        # Keep original mask
        mask = x[..., 9:10]
        x = torch.cat([warped_xyz, vel, acc, mask], dim=-1)

    # Step 2: Existing spatial augmentations
    # ... rotation, scale, noise ...

    return x
```

**Expected Impact:** 5-10% WER reduction
**Risk Level:** MEDIUM - Must verify kinematics recomputation

---

### 3.3 Segment Boundary Jitter (LOW-MEDIUM)

```python
def jitter_boundaries(clip, jitter_frames=3):
    """Add random jitter to clip boundaries, then resample to 32."""
    start_jitter = random.randint(-jitter_frames, jitter_frames)
    end_jitter = random.randint(-jitter_frames, jitter_frames)

    if start_jitter > 0:
        clip = clip[start_jitter:]
    elif start_jitter < 0:
        pad = np.tile(clip[0:1], (-start_jitter, 1, 1))
        clip = np.concatenate([pad, clip], axis=0)

    if end_jitter > 0:
        pad = np.tile(clip[-1:], (end_jitter, 1, 1))
        clip = np.concatenate([clip, pad], axis=0)
    elif end_jitter < 0:
        clip = clip[:end_jitter]

    # Resample back to 32 frames
    if clip.shape[0] != 32:
        clip = temporal_resample(clip, 32)

    return clip
```

**Expected Impact:** 2-3% WER reduction
**Risk Level:** LOW

---

## Phase 4: Inference Enhancements (Day 7-9)

### 4.1 N-gram Language Model for CTC Beam Search

**Critique's caveat:** LM must not be trained only on templates or it will "correct" valid but rare sequences.

```python
# src/build_language_model.py

from collections import defaultdict
import pickle
import math

class GlossNGramLM:
    """Bigram LM for CTC beam rescoring."""

    def __init__(self, smoothing=0.1):
        self.unigram_counts = defaultdict(int)
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.smoothing = smoothing
        self.vocab = set()
        self.total_unigrams = 0

    def train(self, gloss_sequences):
        """Train on list of gloss sequences."""
        for seq in gloss_sequences:
            seq = ['<s>'] + list(seq) + ['</s>']
            for i, gloss in enumerate(seq):
                self.vocab.add(gloss)
                self.unigram_counts[gloss] += 1
                self.total_unigrams += 1
                if i > 0:
                    self.bigram_counts[seq[i-1]][gloss] += 1
        self.vocab_size = len(self.vocab)

    def log_prob(self, gloss, prev_gloss='<s>'):
        """Log probability with smoothing."""
        bigram_count = self.bigram_counts[prev_gloss][gloss]
        prev_count = self.unigram_counts[prev_gloss]

        if prev_count == 0:
            prob = (self.unigram_counts[gloss] + self.smoothing) / \
                   (self.total_unigrams + self.smoothing * self.vocab_size)
        else:
            prob = (bigram_count + self.smoothing) / \
                   (prev_count + self.smoothing * self.vocab_size)

        return math.log(prob + 1e-10)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        lm = cls()
        lm.__dict__.update(data)
        return lm
```

**Integration in `camera_inference.py`:**
- LM weight: Start with `0.3`, tune based on WER
- Use with beam width 25

**Expected Impact:** 5-15% WER reduction
**Risk Level:** MEDIUM - Requires tuning

---

### 4.2 Confidence Threshold

```python
# camera_inference.py

MIN_CONFIDENCE_THRESHOLD = 0.15

def run_full_pipeline(xyz_seq, l_ever, r_ever, ...):
    glosses, confidence, n_signs = run_stage2_recognition(...)

    if confidence < MIN_CONFIDENCE_THRESHOLD:
        return {
            'glosses': [],
            'english': "[Low confidence - please sign again]",
            'confidence': confidence,
            'rejected': True
        }

    english = run_stage3_translation(...)
    return {
        'glosses': glosses,
        'english': english,
        'confidence': confidence,
        'rejected': False
    }
```

---

## Phase 5: Conversational Features (Day 9-12)

**This is what separates "translator" from "conversationalist" for capstone.**

### 5.1 Dialogue Context Window (CRITICAL for Capstone)

```python
# camera_inference.py - Add conversation memory

class ConversationalSLT:
    def __init__(self, s1_model, s2_model, s3_model, s3_tokenizer, ...):
        self.s1_model = s1_model
        self.s2_model = s2_model
        self.s3_model = s3_model
        self.s3_tokenizer = s3_tokenizer

        # Conversation memory
        self.conversation_history = []
        self.max_history_turns = 3

    def translate_with_context(self, glosses):
        """Translate with conversation context."""
        # Build context string
        context_parts = []
        for turn in self.conversation_history[-2:]:
            context_parts.append(f"{turn['speaker']}: {turn['text']}")
        context_str = " | ".join(context_parts)

        # Modified prompt
        if context_str:
            prompt = f"[Context: {context_str}] translate ASL to English: {' '.join(glosses)}"
        else:
            prompt = f"translate ASL to English: {' '.join(glosses)}"

        # Generate
        inputs = self.s3_tokenizer(prompt, return_tensors="pt", max_length=64, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.s3_model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
        )
        translation = self.s3_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Update history
        self.conversation_history.append({
            'speaker': 'User',
            'glosses': glosses,
            'text': translation
        })

        # Trim history
        if len(self.conversation_history) > self.max_history_turns:
            self.conversation_history = self.conversation_history[-self.max_history_turns:]

        return translation

    def add_system_response(self, response_text):
        """Add system's response to history for context."""
        self.conversation_history.append({
            'speaker': 'System',
            'text': response_text
        })

    def clear_conversation(self):
        """Reset conversation (on GOODBYE, STOP, etc.)."""
        self.conversation_history = []
```

### 5.2 Dialogue-Aware Stage 3 Training Data

Create paired dialogue examples:

```csv
dialogue_id,turn_num,gloss,text,speaker
1,1,"HELLO HOW YOU","Hello, how are you?",A
1,2,"I GOOD THANK YOU","I'm doing well, thanks for asking.",B
1,3,"NAME WHAT YOU","What's your name?",A
1,4,"MY NAME J-O-H-N","My name is John.",B
2,1,"HELP ME","Can you help me?",A
2,2,"YES WHAT NEED","Yes, what do you need?",B
```

---

## Phase 6: Validation & Polish (Day 12-14)

### 6.1 Comprehensive Testing

1. **Isolated Sign Accuracy (Stage 1)**
   - Test on held-out set
   - Target: 90%+ top-1 accuracy

2. **Continuous Recognition WER (Stage 2)**
   - Test on real continuous video clips
   - Target: <40% WER (down from ~60%)

3. **Translation Quality (Stage 3)**
   - BLEU score on test set
   - Target: 75+ BLEU (up from ~70)

4. **End-to-End Demo**
   - Real-time camera conversation
   - Multiple turn dialogue

### 6.2 Hyperparameter Tuning

| Parameter | Start | Range | Tune For |
|-----------|-------|-------|----------|
| LM weight | 0.3 | 0.1-0.5 | WER |
| Transition probability | 0.35 | 0.2-0.5 | WER |
| Speed warp range | 0.75-1.25 | 0.7-1.3 | Robustness |
| Label smoothing | 0.10 | 0.05-0.15 | Calibration |
| Beam width | 25 | 10-50 | Latency vs accuracy |

---

## Implementation Priority Matrix

| Priority | Task | Impact | Effort | Phase |
|----------|------|--------|--------|-------|
| **P0** | Fix Stage 3 dataset (questions, single-words) | HIGH | MEDIUM | 2 |
| **P0** | Transition frame injection (Stage 2) | HIGH | MEDIUM | 3 |
| **P1** | MediaPipe confidence alignment | MEDIUM | LOW | 1 |
| **P1** | Temporal speed augmentation | MEDIUM | MEDIUM | 3 |
| **P1** | Dialogue context window | HIGH | MEDIUM | 5 |
| **P2** | N-gram LM for beam search | MEDIUM | MEDIUM | 4 |
| **P2** | Boundary jitter | LOW | LOW | 3 |
| **P3** | Hand dropout augmentation | LOW | LOW | 3 |
| **P3** | Curriculum learning | LOW | MEDIUM | - |

---

## Expected Cumulative Results

| Metric | Current | After Fixes | Improvement |
|--------|---------|-------------|-------------|
| Stage 1 Accuracy | ~85% | ~90% | +5% |
| Stage 2 WER | ~60% | ~35-40% | -25% |
| Stage 3 BLEU | ~70 | ~80 | +10 |
| **E2E Usability** | ~40% | ~65-70% | +25-30% |

---

## Critical Constraints (From Accuracy Critique)

1. **Temporal augmentation MUST recompute kinematics** from warped XYZ. Never warp the 10-channel tensor directly.

2. **Mirror augmentation MUST swap hand indices** (0-20 <-> 21-41) along with X-flip to maintain hand identity.

3. **Transition injection MUST maintain** the clip/target alignment for CTC loss.

4. **LM for beam search** must use diverse training data, not just Stage 3 templates, to avoid over-correcting valid sequences.

5. **Stage 2 sequence length changes** must verify `T % 32 == 0` or handle variable lengths properly in forward().

---

## Files to Modify

| File | Changes |
|------|---------|
| `extract.py` | MediaPipe confidence (0.65 -> 0.80) |
| `train_stage_1.py` | Label smoothing (0.05 -> 0.10), add `online_augment_with_speed()` |
| `train_stage_2.py` | Add `_create_transition_frames()`, boundary jitter, speed augmentation |
| `train_stage_3.py` | Load dialogue-aware dataset, context-aware prompts |
| `camera_inference.py` | Add `ConversationalSLT` class with dialogue history |
| `generate_stage3_data_v2.py` | **NEW** - Enhanced dataset generator |
| `build_language_model.py` | **NEW** - N-gram LM builder |

---

## Success Criteria for Capstone

1. **Live Demo:** 3+ turn conversation with context-aware responses
2. **Translation Quality:** Handles questions, single words, and multi-turn naturally
3. **Robustness:** Works with different signing speeds, occasional occlusion
4. **User Experience:** Confidence feedback, conversation reset on GOODBYE

---

*This plan synthesizes the best recommendations from all three analyses while respecting the accuracy constraints identified in the critique.*
