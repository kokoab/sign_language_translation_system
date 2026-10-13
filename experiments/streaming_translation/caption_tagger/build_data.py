"""Corpus -> per-gloss edit labels; inventory and lookahead statistics (part A).

Writes data.json (vocabularies + encoded rows) and corpus_stats.json.
Only the corpus train split builds inventories; validation is encoded with them.
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from align import corpus_rows, row_sentences, labels, bag_matches, DROP  # noqa: E402
from forms import tokenize, FORMS, GLOSSES  # noqa: E402

WH = {'WHY', 'WHERE', 'WHEN', 'WHAT', 'WHO', 'HOW'}
TIME = {'TOMORROW', 'NOW', 'MORNING', 'NIGHT', 'YESTERDAY', 'TIME', 'WEEK', 'DAY'}
MIN_COUNT = 20
IGNORE = -100


def row_labels(r):
    """Per-gloss labels for a row; None label where the sentence is not monotone."""
    out, kinds = [], []
    for g, s in row_sentences(r):
        toks = tokenize(s)
        lab, m, _ = labels(g, toks)
        b = bag_matches(g, toks)
        if m == b:
            kind = 'monotone' if m == len(g) else 'monotone_with_drop'
            out.extend(lab)
        else:
            d = {gg for gg, l in zip(g, lab) if l['form'] == DROP}
            kind = 'reorder_wh' if d & WH else 'reorder_neg' if 'NO' in d else \
                'reorder_time' if d <= TIME else 'reorder_other'
            out.extend([None] * len(g))
        kinds.append((kind, len(g)))
    return out, kinds


def main():
    rows = {'train': [], 'validation': []}
    kinds = collections.Counter()
    pre_c, post_c, form_c = collections.Counter(), collections.Counter(), collections.Counter()
    tok_total = tok_inserted = 0
    for r in corpus_rows():
        lab, k = row_labels(r)
        rows[r['split']].append((r, lab))
        if r['split'] != 'train':
            continue
        for kind, _ in k:
            kinds[kind] += 1
        for l in lab:
            if l is None:
                continue
            pre_c[l['pre']] += 1; post_c[l['post']] += 1; form_c[l['form']] += 1
            ins = len(l['pre'].split()) + len(l['post'].split())
            tok_inserted += ins
            tok_total += ins + (0 if l['form'] == DROP else len(l['form'].split()))
    pre_v = [p for p, c in pre_c.most_common() if c >= MIN_COUNT]
    post_v = [p for p, c in post_c.most_common() if c >= MIN_COUNT]
    form_v = sorted({' '.join(f) for fs in FORMS.values() for f in fs} | {DROP})
    labelled = sum(pre_c.values())

    def cov(counter, vocab):
        return sum(counter[v] for v in vocab) / sum(counter.values())

    word_ins = collections.Counter()
    for p, c in pre_c.items():
        for w in p.split():
            if w not in '.,?!;:':
                word_ins[w] += c
    stats = {
        'sentences_train': sum(kinds.values()),
        'sentence_kinds': dict(kinds),
        'labelled_gloss_positions': labelled,
        'english_tokens_in_labelled': tok_total,
        'inserted_token_share': tok_inserted / tok_total,
        'pre_distinct': len(pre_c), 'pre_kept': len(pre_v), 'pre_coverage': cov(pre_c, pre_v),
        'pre_empty_share': pre_c[''] / labelled,
        'post_distinct': len(post_c), 'post_kept': len(post_v), 'post_coverage': cov(post_c, post_v),
        'pre_top30': pre_c.most_common(30), 'post_top20': post_c.most_common(20),
        'inserted_word_types': len(word_ins),
        'inserted_words_top40': word_ins.most_common(40),
        'top_n_word_coverage': {n: sum(c for _, c in word_ins.most_common(n)) / sum(word_ins.values())
                                for n in (10, 20, 30, 50, 100)},
        'drop_share': form_c[DROP] / labelled,
    }
    gid = {g: i + 1 for i, g in enumerate(GLOSSES)}  # 0 = pad
    pid, qid, fid = ({v: i for i, v in enumerate(x)} for x in (pre_v, post_v, form_v))

    def enc(r, lab):
        x = [gid[g] for g in r['glosses']]
        pre, form, post = [], [], []
        for g, l in zip(r['glosses'], lab):
            if l is None:
                pre.append(IGNORE); form.append(IGNORE); post.append(IGNORE); continue
            pre.append(pid.get(l['pre'], IGNORE)); post.append(qid.get(l['post'], IGNORE))
            form.append(fid[l['form']])
        return {'x': x, 'pre': pre, 'form': form, 'post': post}

    data = {'glosses': GLOSSES, 'pre_vocab': pre_v, 'post_vocab': post_v, 'form_vocab': form_v,
            'legal_forms': {g: [fid[' '.join(f)] for f in FORMS[g]] + [fid[DROP]] for g in GLOSSES},
            'train': [enc(r, l) for r, l in rows['train']],
            'validation': [enc(r, l) for r, l in rows['validation']]}
    (HERE / 'data.json').write_text(json.dumps(data))
    (HERE / 'corpus_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if not k.endswith(('top30', 'top20', 'top40'))}, indent=1))
    print('pre top', stats['pre_top30'][:30])
    print('post top', stats['post_top20'])
    print('inserted words', stats['inserted_words_top40'])


if __name__ == '__main__':
    main()
