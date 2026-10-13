"""Live English captions from the committed gloss stream (desktop Live screen).

Runtime copy of the experiment engine in
`experiments/streaming_translation/caption_tagger/live_caption.py` (evidence:
`REPORT_V2.md` there). `test/test_live_captions_v17.py` checks that both give
identical captions. Models: `artifacts/models/live_captions_v17_20261014/`.

Behavior while signing:
  * each committed sign appears at once, drafted without guessed punctuation;
  * it is fixed when the next sign arrives (one-sign lookahead);
  * a clause is rewritten once when it closes, only to fix meaning (move a WH
    sign or NO, restore negation, or a tense contradicting its time sign);
  * after `end_seconds` with no sign the tail is finalized; signing again reopens it.
Signs recognized with confidence >= 0.6 are never dropped; words for signs that
were not signed are never added; pronoun/here/there insertions need >= 50%
model probability. Cost is a few milliseconds per sign on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / 'artifacts/models/live_captions_v17_20261014'
PAD, EOS, MAXLEN = 0, 101, 48
# CONTEXT: earlier signs the live tagger sees besides the open clause. 0 was chosen on the
# corpus validation split (sentence-count agreement 75.0% vs 70.3% with 16; same chrF++).
KEEP_CONF, PRON_P, CONTEXT = 0.6, 0.5, 0
PAST = {'went', 'came', 'ate', 'drank', 'gave', 'saw', 'knew', 'took', 'made', 'told', 'thought', 'wrote',
        'found', 'felt', 'slept', 'heard', 'understood', 'was', 'were', 'did', 'had', 'loved', 'liked', 'wanted',
        'needed', 'helped', 'asked', 'worked', 'learned', 'talked', 'stopped', 'tried', 'used', 'waited',
        'listened', 'answered', 'signed'}
WH_SIGNS = {'WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW'}
PRONOUNS = {'me', 'him', 'her', 'it', 'them', 'us', 'he', 'she', 'i', 'my', 'you', 'they', 'we', 'here', 'there'}
PUNCT = '.,!?;:'


def bucket(score):
    return 1 if score >= .6 else 2 if score >= .4 else 3


class Tagger(nn.Module):
    """Same layout as the experiment's tagger.Tagger (state dicts load unchanged)."""

    def __init__(self, n_pre, n_form, n_post, k, d=192, layers=3, heads=4):
        super().__init__()
        self.k = k
        self.tok = nn.Embedding(102, d, padding_idx=PAD)
        self.conf = nn.Embedding(4, d, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN + 1, d)
        self.off = nn.Embedding(8, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mix = nn.Sequential(nn.Linear(3 * d, d), nn.GELU())
        self.pre, self.form, self.post = nn.Linear(d, n_pre), nn.Linear(d, n_form), nn.Linear(d, n_post)

    def forward(self, x, conf, lengths):
        B, T = x.shape
        pos = torch.arange(T)
        h = self.tok(x) + self.conf(conf) + self.pos(pos)[None]
        mask = None if self.k == 'full' else torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
        h = self.norm(self.enc(h, mask=mask, src_key_padding_mask=x == PAD))
        j = pos[None].expand(B, T) if self.k == 'full' else torch.minimum(pos[None] + self.k, lengths[:, None])
        hj = torch.gather(h, 1, j[..., None].expand(B, T, h.shape[-1]))
        z = self.mix(torch.cat([h, hj, self.off((j - pos[None]).clamp(0, 7))], -1))
        return self.pre(z), self.form(z), self.post(z)


class Mover(nn.Module):
    """Same layout as the experiment's finalizer.Mover."""

    def __init__(self, d=160, layers=3, heads=4):
        super().__init__()
        self.tok = nn.Embedding(102, d, padding_idx=PAD)
        self.conf = nn.Embedding(4, d, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN + 1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.src = nn.Linear(d, 1)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)

    def forward(self, x, conf):
        T = x.shape[1]
        h = self.tok(x) + self.conf(conf) + self.pos(torch.arange(T))[None]
        h = self.norm(self.enc(h, src_key_padding_mask=x == PAD))
        pad = x == PAD
        return h, self.src(h).squeeze(-1).masked_fill(pad, -1e4), pad

    def dst_logits(self, h, pad, src_idx):
        b = torch.arange(h.shape[0])
        return torch.einsum('bd,btd->bt', self.q(h[b, src_idx]), self.k(h)).masked_fill(pad, -1e4)


def tokens_of(tags):
    out = []
    for pre, form, post in tags:
        out += pre.split() + ([] if form == '<drop>' else form.split()) + post.split()
    return out


def detokenize(tokens):
    out, cap = [], True
    for tok in tokens:
        if tok in PUNCT or tok == "'s":
            if not out:
                continue  # orphan punctuation (e.g. a clause whose low-confidence signs were all dropped)
            if tok in PUNCT and out[-1][-1:] in PUNCT:
                continue
            if out:
                out[-1] += tok
            else:
                out.append(tok)
            if tok in '.!?':
                cap = True
            continue
        w = 'FS0' if tok == 'fs0' else ('I' if tok == 'i' else tok)
        if cap and w:
            w = w[0].upper() + w[1:]
        cap = False
        out.append(w)
    return ' '.join(out)


def merge_fingerspelling(glosses, scores, known):
    """fs-GE fs-LO -> one name slot 'Gelo'; an unknown gloss is shown as a name too."""
    g2, s2, names, prev_fs = [], [], [], False
    for g, s in zip(glosses, scores):
        spelled = g.startswith('fs-')
        if spelled and g2 and g2[-1] == 'FS0' and prev_fs:
            names[-1] += g[3:].lower(); s2[-1] = max(s2[-1], s)
        elif spelled or g not in known:
            g2.append('FS0'); s2.append(s)
            names.append(g[3:].title() if spelled else g.title())
        else:
            g2.append(g); s2.append(s)
        prev_fs = spelled
    return g2, s2, names


def restore(text, names):
    for n in names:
        text = text.replace('FS0', n, 1)
    return text


def apply_move(seq, src, dst):
    item = seq[src]
    out = [x for i, x in enumerate(seq) if i != src]
    out.insert(dst if dst <= src else dst - 1, item)
    return out


def meaning_key(toks):
    t = set(toks)
    future, past = 'will' in t or 'going' in t, bool(t & PAST)
    return (bool(t & {'not', 'never'}), '?' in toks,
            'past' if past and not future else 'future' if future and not past else 'present')


def tense_conflict(toks):
    """A verb tense contradicting its governing time word (time words govern what
    follows them; words before the first time word belong to it)."""
    times = [(i, t) for i, t in enumerate(toks) if t in ('tomorrow', 'yesterday')]
    if not times:
        return False
    for i, t in enumerate(toks):
        if t in PAST or t == 'will':
            before = [x for x in times if x[0] < i]
            gov = before[-1][1] if before else times[0][1]
            if (t in PAST and gov == 'tomorrow') or (t == 'will' and gov == 'yesterday'):
                return True
    return False


def needs_revision(live, fin, moved):
    if moved in WH_SIGNS or moved == 'NO':
        return True
    if meaning_key(fin)[0] and not meaning_key(live)[0]:
        return True
    return tense_conflict(live) and not tense_conflict(fin)


class CaptionModels:
    """Loads the packaged models once; shared by all caption sessions."""

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        self.manifest = json.loads((model_dir / 'manifest.json').read_text())
        V = self.vocab = json.loads((model_dir / 'vocab.json').read_text())
        self.gid = {g: i + 1 for i, g in enumerate(V['glosses'])}
        self.known = set(V['glosses']) - {'FS0'}
        n = (len(V['pre_vocab']), len(V['form_vocab']), len(V['post_vocab']))
        lm = torch.full((102, n[1]), -1e4)
        lm[PAD] = 0; lm[EOS] = 0
        for i, g in enumerate(V['glosses']):
            lm[i + 1, V['legal_forms'][g]] = 0
        self.lm = lm
        pm = torch.full((n[0], n[1]), -1e4)
        for p, f in V['allowed_pairs']:
            pm[p, f] = 0.
        self.pair_mask = pm
        self.drop = V['form_vocab'].index('<drop>')
        self.pre_pron = torch.tensor([bool(set(p.split()) & PRONOUNS) for p in V['pre_vocab']])
        self.post_pron = torch.tensor([bool(set(p.split()) & PRONOUNS) for p in V['post_vocab']])

        def load(cls, name, *a):
            m = cls(*a)
            m.load_state_dict(torch.load(model_dir / name, map_location='cpu', weights_only=True))
            return m.eval()

        self.live = [load(Tagger, p, *n, 1) for p in self.manifest['live_models']]
        self.draft = load(Tagger, self.manifest['draft_model'], *n, 0)
        self.labeler = load(Tagger, self.manifest['labeler'], *n, 'full')
        self.mover = load(Mover, self.manifest['mover'])
        self._final = {}

    @staticmethod
    def _guard(lp, is_pron):
        best = lp.argmax()
        if is_pron[best] and lp[best].exp() < PRON_P:
            return lp.masked_fill(is_pron, -1e4)
        return lp

    @torch.inference_mode()
    def tags(self, models, glosses, scores, ended):
        V = self.vocab
        models = models if isinstance(models, (list, tuple)) else [models]
        n = len(glosses)
        xt = torch.tensor([[self.gid[g] for g in glosses] + ([EOS] if ended else [])])
        ct = torch.tensor([[bucket(s) for s in scores] + ([1] if ended else [])])
        L = torch.tensor([n if ended else n - 1])
        outs = [m(xt, ct, L) for m in models]
        lp_pre = sum(o[0].log_softmax(-1)[0] for o in outs) / len(outs)
        lp_post = sum(o[2].log_softmax(-1)[0] for o in outs) / len(outs)
        lp_form = sum((o[1] + self.lm[xt]).log_softmax(-1)[0] for o in outs) / len(outs)
        result = []
        for i in range(n):
            f = lp_form[i].clone()
            if scores[i] >= KEEP_CONF:
                f[self.drop] = -1e4
            p = self._guard(lp_pre[i], self.pre_pron)
            q = self._guard(lp_post[i], self.post_pron)
            joint = p[:, None] + f[None, :] + self.pair_mask
            pi, fi = divmod(int(joint.argmax()), joint.shape[1])
            result.append((V['pre_vocab'][pi], V['form_vocab'][fi], V['post_vocab'][int(q.argmax())]))
        return result

    @torch.inference_mode()
    def finalize(self, glosses, scores):
        """Clause rewrite: at most one sign moved, then full-context rendering."""
        key = json.dumps([glosses, [bucket(s) for s in scores]])
        if key in self._final:
            return self._final[key]
        n = len(glosses)
        x = torch.tensor([[self.gid[g] for g in glosses] + [EOS]])
        c = torch.tensor([[bucket(s) for s in scores] + [1]])
        h, sl, pad = self.mover(x, c)
        src, moved, g2, s2 = int(sl[0].argmax()), None, list(glosses), list(scores)
        if src < n:
            dst = int(self.mover.dst_logits(h, pad, torch.tensor([src]))[0].argmax())
            g2, s2 = apply_move(glosses, src, dst), apply_move(scores, src, dst)
            moved = glosses[src] if g2 != list(glosses) else None
        toks = tokens_of(self.tags(self.labeler, g2, s2, True))
        if not re.search(r'[.?!]$', ' '.join(toks)):
            toks.append('.')
        self._final[key] = (toks, moved)
        return toks, moved


class CaptionEngine:
    """Stateless view computation over a gloss prefix (mirrors the experiment's LiveCaptioner.step)."""

    def __init__(self, models: CaptionModels):
        self.m = models

    def _close(self, g, s, a, b, tags_slice, final):
        live = tokens_of(tags_slice)
        fin, moved = self.m.finalize(g[a:b], s[a:b])
        if needs_revision(live, fin, moved):
            return fin
        if final and fin and fin[-1] in '.?!' and not (live and live[-1] in '.?!'):
            return live + [fin[-1]]
        return live

    def step(self, state, g, s, ended, final=True, tail_period=True, use_draft=True):
        n, o = len(g), state['open']
        c0 = max(0, o - CONTEXT)
        tags = self.m.tags(self.m.live, g[c0:], s[c0:], ended)
        n_commit = n if ended else max(o, n - 1)
        frozen, start, toks = list(state['frozen']), o, []
        for i in range(o, n_commit):
            if re.search(r'[.?!]', tags[i - c0][2]) or (ended and i == n - 1):
                clause = self._close(g, s, start, i + 1, tags[start - c0:i + 1 - c0],
                                     final=final or not (ended and i == n - 1))
                if ended:
                    toks += clause
                else:
                    frozen += clause
                start = i + 1
        if not ended:
            state['frozen'], state['open'] = frozen, start
            toks = []
        toks += tokens_of(tags[start - c0:n_commit - c0])
        if not ended and use_draft and n_commit < n:
            dtags = self.m.tags(self.m.draft, g[c0:], s[c0:], False)[n_commit - c0:]
            toks += tokens_of([(p, f, '') for p, f, _ in dtags])
        if ended and not tail_period and toks and toks[-1] == '.':
            toks = toks[:-1]
        return frozen + toks


@dataclass
class CaptionUpdate:
    text: str            # full caption to display
    speak: str = ''      # newly settled English to speak (may be empty)
    kind: str = 'sign'   # 'sign' | 'end_of_turn' | 'finish'


class LiveCaptions:
    """One conversation's captions. Feed committed signs; call tick() every frame."""

    def __init__(self, models: CaptionModels | None = None, end_seconds: float = 3.5,
                 freeze_seconds: float | None = None):
        """end_seconds: pause after which the tail is settled and spoken.
        freeze_seconds: a longer pause after which that settled tail becomes final, so the
        next sign starts a new sentence (None: signing again always reopens the tail)."""
        self.models = models or CaptionModels()
        self.engine = CaptionEngine(self.models)
        self.end_seconds, self.freeze_seconds = end_seconds, freeze_seconds
        self.reset()

    def reset(self):
        self.prefix = ''     # settled text before an interjection (shown before the current caption)
        self._new_utterance()

    def _new_utterance(self):
        self.glosses, self.scores = [], []
        self.state = {'frozen': [], 'open': 0}
        self.tokens, self.names, self.said = [], [], []   # said: display tokens already spoken
        self.last_sign_at, self.ended = None, False
        self.text = ''

    def _speech(self, upto):
        """Speak settled words from where the spoken text and the caption first differ: new
        words normally, or the corrected part if already-spoken words were revised."""
        upto = min(upto, len(self.tokens))
        start = 0
        for a, b in zip(self.said, self.tokens[:upto]):
            if a != b:
                break
            start += 1
        if upto <= start:
            return ''
        offset = sum(t == 'fs0' for t in self.tokens[:start])  # names already spoken
        words = restore(detokenize(self.tokens[start:upto]), self.names[offset:])
        self.said = list(self.tokens[:upto])
        return words if re.search(r'\w', words) else ''

    def _view(self, ended, final=True, tail_period=True):
        g, s, self.names = merge_fingerspelling(self.glosses, self.scores, self.models.known)
        self.tokens = self.engine.step(self.state, g, s, ended, final=final, tail_period=tail_period)
        self.text = ' '.join(x for x in (self.prefix, restore(detokenize(self.tokens), self.names)) if x)

    def push(self, gloss: str, score: float, seconds: float) -> CaptionUpdate:
        if (self.ended and self.freeze_seconds is not None
                and seconds - self.last_sign_at >= self.freeze_seconds - 1e-6):
            # Long silence: the settled tail is final; this sign starts a new sentence.
            n = len(merge_fingerspelling(self.glosses, self.scores, self.models.known)[0])
            final = list(self.tokens)
            if final and final[-1] not in '.?!':
                final.append('.')
            self.state = {'frozen': final, 'open': n}
        self.glosses.append(gloss); self.scores.append(float(score))
        self.last_sign_at, self.ended = seconds, False
        self._view(False)
        return CaptionUpdate(self.text, self._speech(len(self.state['frozen'])), 'sign')

    @property
    def spoken(self):
        return len(self.said)

    def tick(self, seconds: float) -> CaptionUpdate | None:
        """End of turn: no new sign for end_seconds finalizes the tail (provisionally)."""
        if self.ended or self.last_sign_at is None or seconds - self.last_sign_at < self.end_seconds - 1e-6:
            return None
        self.ended = True
        self._view(True, final=True, tail_period=False)
        return CaptionUpdate(self.text, self._speech(len(self.tokens)), 'end_of_turn')

    def interject(self, text: str) -> CaptionUpdate:
        """Settle the current caption and append a fixed sentence (e.g. a novelty gesture)."""
        speak = ''
        if self.glosses:
            self._view(True)
            speak = self._speech(len(self.tokens))
        self.prefix = ' '.join(x for x in (self.text, text) if x)
        self._new_utterance()
        self.text = self.prefix
        return CaptionUpdate(self.text, ' '.join(x for x in (speak, text) if x), 'interjection')

    def finish(self) -> CaptionUpdate:
        """Explicit Finish: final caption of the whole utterance; the caller resets afterwards."""
        if not self.glosses:
            return CaptionUpdate(self.text, '', 'finish')
        self._view(True)
        return CaptionUpdate(self.text, self._speech(len(self.tokens)), 'finish')
