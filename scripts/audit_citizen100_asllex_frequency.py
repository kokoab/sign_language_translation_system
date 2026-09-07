"""Position the Citizen-100 vocabulary in the ASL-LEX 2.0 frequency distribution.

Joins active/v17/citizen100_manifest.json (citizen_asl_lex_code) to the official
ASL-LEX 2.0 signdata.csv (Code) and reports where the 100 selected classes sit
relative to all 2,723 rated ASL signs.

Backs artifacts/reports/CITIZEN100_VOCABULARY_JUSTIFICATION.md.
"""
import bisect, csv, json, statistics as st
from collections import Counter

ASLLEX = "data/local/dataset_metadata/asllex2_official/signdata.csv"
MANIFEST = "active/v17/citizen100_manifest.json"


def num(row, key):
    try:
        return float((row.get(key) or "").strip())
    except ValueError:
        return None


rows = list(csv.DictReader(open(ASLLEX, encoding="utf-8", errors="replace")))
freq = {r["Code"]: num(r, "SignFrequency(M)") for r in rows}
ours = {c["citizen_asl_lex_code"]: c["canonical_label"]
        for c in json.load(open(MANIFEST))["classes"]}

missing = [c for c in ours if c not in freq]
assert not missing, f"manifest codes absent from ASL-LEX: {missing}"

allv = sorted(v for v in freq.values() if v is not None)
ourv = [freq[c] for c in ours]
pctile = lambda v: 100.0 * bisect.bisect_left(allv, v) / len(allv)

print(f"ASL-LEX 2.0 : n={len(allv)} mean={st.mean(allv):.2f} sd={st.pstdev(allv):.2f}")
print(f"Citizen-100 : n={len(ourv)} mean={st.mean(ourv):.2f} sd={st.pstdev(ourv):.2f}")
print(f"z of our mean vs ASL-LEX: {(st.mean(ourv)-st.mean(allv))/st.pstdev(allv):+.2f} SD")
ps = sorted(pctile(v) for v in ourv)
print(f"median percentile: {st.median(ps):.1f}")
for thr in (90, 75, 50, 25):
    print(f"  top {100-thr:2d}% of ASL-LEX: {sum(p >= thr for p in ps)}/100")

ranked = [c for _, c in sorted(((v, c) for c, v in freq.items() if v is not None),
                               reverse=True)]
for n in (100, 300, 1000):
    print(f"ASL-LEX top-{n:<4d}: {len(set(ranked[:n]) & set(ours))} in Citizen-100")

for col in ("Phonological Complexity", "Neighborhood Density 2.0",
            "SignDuration(ms)", "Iconicity(M)"):
    a = [v for r in rows if r["Code"] in ours and (v := num(r, col)) is not None]
    b = [v for r in rows if r["Code"] not in ours and (v := num(r, col)) is not None]
    print(f"{col:28s} ours={st.mean(a):8.2f}  rest={st.mean(b):8.2f}")

print("LexicalClass:", Counter(r["LexicalClass"] for r in rows if r["Code"] in ours))
