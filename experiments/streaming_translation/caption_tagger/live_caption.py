"""Live caption engine: append-only tagger + learned clause finalizer.

Per recognized sign:
  * draft: the newest sign is shown at once with the k=0 tagger;
  * commit: sign i is fixed when sign i+1 arrives (k=1 tagger);
  * clause close: when a committed sign ends a sentence, the clause is revised
    once by the finalizer (learned move of at most one sign + full-context render);
  * pause: if no sign arrives for `pause_s`, the open clause is rendered as if the
    utterance ended and finalized provisionally; a later sign reopens it.
Decoding (same for all models): a sign recognized with confidence >= 0.6 is never
dropped; an insertion containing a pronoun is used only when the model gives that
insertion >= 0.5 probability (otherwise the best pronoun-free insertion);
consecutive fingerspelled fragments form one name.
"""
import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from forms import detokenize  # noqa: E402
from tagger import Tagger, legal_mask, bucket, EOS  # noqa: E402
from finalizer import Mover, apply_move  # noqa: E402

PAST = {'went', 'came', 'ate', 'drank', 'gave', 'saw', 'knew', 'took', 'made', 'told', 'thought', 'wrote',
        'found', 'felt', 'slept', 'heard', 'understood', 'was', 'were', 'did', 'had', 'loved', 'liked', 'wanted',
        'needed', 'helped', 'asked', 'worked', 'learned', 'talked', 'stopped', 'tried', 'used', 'waited',
        'listened', 'answered', 'signed'}


def meaning_key(toks):
    """What a clause revision is allowed to change: polarity, question, tense."""
    t = set(toks)
    future = 'will' in t or 'going' in t
    past = bool(t & PAST)
    return (bool(t & {'not', 'never'}), '?' in toks, 'past' if past and not future else
            'future' if future and not past else 'present')


# Insertions that can add a referent the signer did not sign; used only when confident.
WH_SIGNS = {'WHAT', 'WHERE', 'WHEN', 'WHO', 'WHY', 'HOW'}


def tense_conflict(glosses, toks):
    """A verb tense that contradicts its governing time word. A time word governs
    the words after it until the next time word (ASL time-first order); words
    before the first time word belong to the first one (time-final sentences)."""
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


def needs_revision(glosses, live, fin, moved):
    lk, fk = meaning_key(live), meaning_key(fin)
    if moved in WH_SIGNS or moved == 'NO':
        return True
    if fk[0] and not lk[0]:  # negation would otherwise be missing
        return True
    # tense: only when the live text contradicts a time sign and the revision does not
    return tense_conflict(glosses, live) and not tense_conflict(glosses, fin)


PRONOUNS = {'me', 'him', 'her', 'it', 'them', 'us', 'he', 'she', 'i', 'my', 'you', 'they', 'we', 'here', 'there'}
KEEP_CONF = 0.6
PRON_P = 0.5


class Decoder:
    """Shared vocabularies, masks and the decoding rules above."""

    def __init__(self):
        D = self.D = json.loads((HERE / 'data.json').read_text())
        self.gid = {g: i + 1 for i, g in enumerate(D['glosses'])}
        self.lm = legal_mask(D, 'cpu')
        self.drop = D['form_vocab'].index('<drop>')
        pairs = torch.zeros(len(D['pre_vocab']), len(D['form_vocab']))
        for r in D['train']:
            for p, f in zip(r['pre'], r['form']):
                if p != -100 and f != -100:
                    pairs[p, f] += 1
        self.pair_mask = torch.where(pairs >= 5, 0.0, -1e4)
        self.pre_pron = torch.tensor([bool(set(p.split()) & PRONOUNS) for p in D['pre_vocab']])
        self.post_pron = torch.tensor([bool(set(p.split()) & PRONOUNS) for p in D['post_vocab']])

    def model(self, k, path):
        D = self.D
        m = Tagger(len(D['pre_vocab']), len(D['form_vocab']), len(D['post_vocab']), k)
        m.load_state_dict(torch.load(HERE / path, map_location='cpu')); m.eval()
        return m

    @staticmethod
    def _pron_guard(lp, is_pron):
        """Drop pronoun-bearing classes unless the best one is confident."""
        best = lp.argmax()
        if is_pron[best] and lp[best].exp() < PRON_P:
            return lp.masked_fill(is_pron, -1e4)
        return lp

    @torch.no_grad()
    def tags(self, model, glosses, scores, ended, lookahead_index=True):
        x = [self.gid[g] for g in glosses]
        conf = [bucket(s) for s in scores]
        n = len(x)
        xt = torch.tensor([x + ([EOS] if ended else [])])
        ct = torch.tensor([conf + ([1] if ended else [])])
        L = torch.tensor([n if ended else n - 1])
        models = model if isinstance(model, (list, tuple)) else [model]
        outs = [m(xt, ct, L) for m in models]  # several models = averaged log-probabilities
        lp_pre = sum(o[0].log_softmax(-1)[0] for o in outs) / len(outs)
        lp_post = sum(o[2].log_softmax(-1)[0] for o in outs) / len(outs)
        lp_form = sum((o[1] + self.lm[xt]).log_softmax(-1)[0] for o in outs) / len(outs)
        out = []
        for i in range(n):
            f = lp_form[i].clone()
            if scores[i] >= KEEP_CONF:
                f[self.drop] = -1e4
            p = self._pron_guard(lp_pre[i], self.pre_pron)
            q = self._pron_guard(lp_post[i], self.post_pron)
            joint = p[:, None] + f[None, :] + self.pair_mask
            pi, fi = divmod(int(joint.argmax()), joint.shape[1])
            out.append((self.D['pre_vocab'][pi], self.D['form_vocab'][fi], self.D['post_vocab'][int(q.argmax())]))
        return out


def tokens_of(tags):
    toks = []
    for pre, form, post in tags:
        toks += pre.split() + ([] if form == '<drop>' else form.split()) + post.split()
    return toks


def merge_fingerspelling(glosses, scores):
    """fs-GE fs-LO -> one FS0 slot named 'Gelo'. Returns glosses, scores, names."""
    g2, s2, names, prev_fs = [], [], [], False
    for g, s in zip(glosses, scores):
        if g.startswith('fs-') and g2 and g2[-1] == 'FS0' and prev_fs:
            names[-1] += g[3:].lower(); s2[-1] = max(s2[-1], s)
        elif g.startswith('fs-'):
            g2.append('FS0'); s2.append(s); names.append(g[3:].title())
        else:
            g2.append(g); s2.append(s)
        prev_fs = g.startswith('fs-')
    return g2, s2, names


def restore(text, names):
    for n in names:
        text = text.replace('FS0', n, 1)
    return text


class Finalizer:
    def __init__(self, dec):
        self.dec = dec
        self.mover = Mover(); self.mover.load_state_dict(torch.load(HERE / 'mover.pt', map_location='cpu'))
        self.mover.eval()
        self.labeler = dec.model('full', 'labeler.pt')
        self.cache = {}

    @torch.no_grad()
    def __call__(self, glosses, scores):
        key = json.dumps([glosses, [bucket(s) for s in scores]])
        if key in self.cache:
            return self.cache[key]
        n = len(glosses)
        x = torch.tensor([[self.dec.gid[g] for g in glosses] + [EOS]])
        c = torch.tensor([[bucket(s) for s in scores] + [1]])
        h, sl, pad = self.mover(x, c)
        src = int(sl[0].argmax())
        moved = None
        if src < n:
            dst = int(self.mover.dst_logits(h, pad, torch.tensor([src]))[0].argmax())
            g2, s2 = apply_move(glosses, src, dst), apply_move(scores, src, dst)
            moved = glosses[src] if g2 != list(glosses) else None
        else:
            g2, s2 = glosses, scores
        toks = tokens_of(self.dec.tags(self.labeler, g2, s2, True))
        if not re.search(r'[.?!]$', ' '.join(toks)):
            toks.append('.')
        self.cache[key] = (toks, moved)
        return toks, moved


class LiveCaptioner:
    """Stateful replay. Closed clauses are frozen; the live tagger sees only the
    open clause plus up to `context` earlier signs, so cost per sign is bounded
    however long the conversation runs."""

    def __init__(self, pause_s=None, end_s=3.5, draft=True, finalize=True, commit_k=1, context=16,
                 live_path=None, revise='meaning', draft_delay=0.0, draft_mode='full'):
        self.dec = Decoder()
        self.k, self.context = commit_k, context
        paths = live_path or [f'tagger_k{commit_k}.pt']
        paths = [paths] if isinstance(paths, str) else paths
        self.live = [self.dec.model(commit_k, p) for p in paths]
        self.draft_model = self.dec.model(0, 'tagger_k0.pt') if draft else None
        self.fin = Finalizer(self.dec) if finalize else None
        self.pause_s, self.end_s, self.revise = pause_s, end_s, revise
        self.draft_delay = draft_delay  # show the newest sign's draft only if no sign follows within this
        self.draft_mode = draft_mode  # 'full' | 'no_post' (no guessed punctuation/tail) | 'form' (bare word)

    def _close(self, g, s, a, b, tags_slice, final=True):
        """Clause text. revise='meaning': keep the live caption unless it has a meaning
        problem the finalizer fixes: a WH sign or NO needs moving, negation would be
        lost, or the live verb tense contradicts a time sign. Cosmetic rewrites are
        skipped (viewers prefer stability). final=False: no punctuation is added."""
        live = tokens_of(tags_slice)
        if not self.fin:
            return live
        fin, moved = self.fin(g[a:b], s[a:b])
        if self.revise == 'always' or needs_revision(g[a:b], live, fin, moved):
            return fin
        if final and fin and fin[-1] in '.?!' and not (live and live[-1] in '.?!'):
            return live + [fin[-1]]
        return live

    def step(self, state, g, s, ended, final=True, tail_period=True, use_draft=True):
        """Display tokens for prefix g; advances frozen clauses in `state` unless ended
        (a pause/end view is provisional and does not freeze anything).
        final=False marks a short-pause view."""
        n, o = len(g), state['open']
        c0 = max(0, o - self.context)
        tags = self.dec.tags(self.live, g[c0:], s[c0:], ended)
        n_commit = n if ended else max(o, n - self.k)
        frozen, start, toks = list(state['frozen']), o, []
        for i in range(o, n_commit):
            t = tags[i - c0]
            if re.search(r'[.?!]', t[2]) or (ended and i == n - 1):
                is_tail = ended and i == n - 1
                clause = self._close(g, s, start, i + 1, tags[start - c0:i + 1 - c0],
                                     final=final or not is_tail)
                if ended:
                    toks += clause
                else:
                    frozen += clause
                start = i + 1
        if not ended:
            state['frozen'], state['open'] = frozen, start
            toks = []
        toks += tokens_of(tags[start - c0:n_commit - c0])
        if not ended and use_draft and self.draft_model is not None and n_commit < n:
            dtags = self.dec.tags(self.draft_model, g[c0:], s[c0:], False)[n_commit - c0:]
            if self.draft_mode == 'no_post':
                dtags = [(p, f, '') for p, f, _ in dtags]
            elif self.draft_mode == 'form':
                dtags = [('', f, '') for _, f, _ in dtags]
            toks += tokens_of(dtags)
        if ended and not tail_period and toks and toks[-1] == '.':
            toks = toks[:-1]  # live tail: a period would be erased if signing resumes
        return frozen + toks

    def run(self, glosses, scores, arrivals, end_at=None):
        """Replay a sign stream; event times include measured compute on a serial queue.
        Updates on every recognized item, including each fingerspelled fragment, so
        a pause is only detected when the signer actually stops.
        end_at: explicit end signal time (None = only pauses close the tail)."""
        t0 = arrivals[0]; ts = [t - t0 for t in arrivals]
        state = {'frozen': [], 'open': 0}
        events, busy, ms_all, prev, erased, erased_words = [], 0.0, [], [], 0, 0
        names = []

        def emit(at, toks, kind):
            nonlocal prev, erased, erased_words
            lcp = 0
            for x, y in zip(prev, toks):
                if x != y:
                    break
                lcp += 1
            erased += len(prev) - lcp
            erased_words += sum(1 for t in prev[lcp:] if t not in '.,!?;:')
            prev = toks
            events.append({'at': round(at, 3), 'kind': kind, 'text': restore(detokenize(toks), names)})

        for j in range(len(glosses)):
            g, s, names = merge_fingerspelling(glosses[:j + 1], scores[:j + 1])
            nxt = ts[j + 1] if j + 1 < len(glosses) else None
            immediate = self.draft_delay == 0
            c = time.perf_counter(); toks = self.step(state, g, s, False, use_draft=immediate)
            ms = (time.perf_counter() - c) * 1000; ms_all.append(ms)
            busy = max(ts[j], busy) + ms / 1000
            emit(busy, toks, 'sign')
            if not immediate and self.draft_delay is not None and (nxt is None or nxt > ts[j] + self.draft_delay):
                c = time.perf_counter(); toks = self.step(state, g, s, False, use_draft=True)
                ms = (time.perf_counter() - c) * 1000; ms_all.append(ms)
                busy = max(ts[j] + self.draft_delay, busy) + ms / 1000
                if toks != prev:
                    emit(busy, toks, 'draft')
            for wait, kind, fin in ((self.pause_s, 'pause', False), (self.end_s, 'end_of_turn', True)):
                if wait is not None and (nxt is None or nxt > ts[j] + wait):
                    c = time.perf_counter(); toks = self.step(state, g, s, True, final=fin, tail_period=False)
                    ms = (time.perf_counter() - c) * 1000; ms_all.append(ms)
                    busy = max(ts[j] + wait, busy) + ms / 1000
                    if toks != prev:
                        emit(busy, toks, kind)
        if end_at is not None:
            toks = self.step(state, g, s, True); emit(max(end_at - t0, busy), toks, 'end')
        return {'events': events, 'final': events[-1]['text'], 'n_final_tokens': len(prev),
                'normalized_erasure': erased / max(1, len(prev)),
                'normalized_word_erasure': erased_words / max(1, sum(t not in '.,!?;:' for t in prev)),
                'compute_ms': ms_all}

    def final_text(self, glosses, scores):
        """Caption after the whole stream and a closing pause (incremental, as live)."""
        g, s, names = merge_fingerspelling(glosses, scores)
        state = {'frozen': [], 'open': 0}
        for j in range(len(g)):
            self.step(state, g[:j + 1], s[:j + 1], False)
        return restore(detokenize(self.step(state, g, s, True)), names)


