# Locked100 continuous phrase discovery: full MoLo annotation audit

Audited 2026-09-07. **Eight short textual candidates; zero approved training spans.**
The locked100 vocabulary is unchanged. This audit does not establish improved
recognition, live speed, or signer generalization.

## Findings

| Check | Result |
| --- | --- |
| Public EAF transcripts downloaded and size-verified | 16; 5,284,280 bytes |
| Manual annotations, counting both hands | 6,878 |
| Exact Citizen raw-label occurrences before deduplication | 551 |
| Empty manual transcripts | 4 |
| Transcripts with at least two manual annotations | 11 |
| Filename-inferred signers in those 11 transcripts | 6, across 3 sessions |
| Conservative contiguous candidate spans | 8; 7 distinct sequences |
| Candidate length | Seven pairs; one triple |
| Candidate raw vocabulary coverage | 11 of 100 |
| Eligible for training | 0 |

These are fragments within conversations, not eight complete sentences. The
collection-wide participant count does not describe its available ID-gloss coverage.
The publisher describes the transcripts as work in progress and reports re-editing
video segments. [MoLo collection](https://ida.gallaudet.edu/molo/)

## Exact candidates

Times below are EAF milliseconds converted to seconds, **not verified video cuts**.
Names are inferred from transcript filenames; identity and split independence remain
unverified. Retain an entire session together when designing future splits, because
participants share recordings.

| Session / task | Filename signer | Start–end (s) | Exact raw labels |
| --- | --- | --- | --- |
| 001 / S | RileySchultz | 94.507–95.394 | WORK SCHOOL |
| 001 / S | RileySchultz | 156.604–157.106 | HAVE MORE |
| 002 / N | KennethDeHaan | 37.491–37.837 | TIME YES |
| 002 / N | KennethDeHaan | 186.008–186.563 | THINK YES |
| 002 / N | TimothyMiller | 154.982–155.785 | YES SEE READ |
| 002 / S | KennethDeHaan | 748.059–748.675 | GOOD YES |
| 002 / S | TimothyMiller | 576.584–577.151 | YES BAD |
| 003 / N | JonHenner | 491.387–492.011 | THINK YES |

## Video availability

The OSF native-storage root and child-node lists are empty, but its linked public
Google Drive provider lists **26 MP4 files and one document**. The saved provider
listing has no next page. This audit used public HTTP metadata, not a signed-in Drive
account. [Public provider listing](https://api.osf.io/v2/nodes/wma3e/files/googledrive/?page%5Bsize%5D=100)

* **One candidate:** the [MoLo003 narrative page](https://ida.gallaudet.edu/molo/1/)
  has the matching recording title and a native download advertised as 1,648.4 MB.
  This establishes a source lead, not matching video bytes or timing.
* **Two candidates:** EAFs reference `MoLo001_S_7_8`, while the
  [current video page](https://ida.gallaudet.edu/molo/9/) lists `MoLo001_S_4_5`
  (487.5 MB). Renaming is insufficient to establish alignment after re-editing.
* **Five candidates:** referenced MoLo002 recordings were absent from the complete
  public OSF provider listing and inspected gallery page. This is not proof that
  they are unavailable through every source.

Direct source links and matching notes are saved in `media_audit.json`. No videos
were downloaded. The publisher licenses videos CC BY-NC-SA 4.0; retain attribution
and those terms in any downstream acquisition. [Publisher terms](https://ida.gallaudet.edu/molo/)

## Screening and remaining gates

`scripts/audit_molo_continuous_v17.py` uses the standard library. It parses timed
manual annotations from both hand tiers, resolves linked notes, and fails on missing
manual timing. It compares exact case-sensitive raw labels to the frozen manifest;
it does not merge aliases or numeric variants. Simultaneous duplicate hand labels
merge only when their nonempty Signbank CVE IDs agree. Other-hand unsupported labels,
conflicting variants, and nonempty linked notes break a run. Sequential repeated
signs remain separate. A maximum 300 ms unannotated gap is a conservative screening
heuristic, not a learned sign boundary. Runs need at least two signs.

ASL Signbank CVE IDs are preserved for review. Matching a written label does **not**
verify the exact Citizen ASL-LEX variant. All candidates explicitly carry
`eligible_for_training: false`: lexical crosswalk, source-video alignment,
signer/session independence, and annotation completeness remain unresolved.

This source does not currently replace the missing 30 recorded phrases. The earlier
[NCSLGR audit](../continuous_reel_v17_web_audit_v1/README.md) has 77 textual leads,
but still needs timed annotations and signer/variant verification. The next useful
data action is resolving those annotation gates; training on these eight fragments
would not substantiate continuous recognition across the locked100 vocabulary.

## Reproduce

```sh
venv/bin/python scripts/audit_molo_continuous_v17.py \
  --annotation-root artifacts/reports/continuous_reel_v17_web_audit_v2/molo_annotations \
  --source-listing artifacts/reports/continuous_reel_v17_web_audit_v1/molo_audit.json \
  --output-root artifacts/reports/continuous_reel_v17_web_audit_v2
venv/bin/python -m unittest test.test_audit_molo_continuous_v17 -v
```

`audit.json` records per-file hashes, source URLs, counts, media names, participant
fields, external vocabulary references, and the frozen manifest hash.
`candidates.json` preserves sign times, annotation IDs, hands, CVE IDs and blockers.
`molo_annotations/` contains all 16 original EAFs. Public source responses are saved
alongside them. Four focused regression tests pass, including two-hand blockers,
repeated signs, strict labels, annotation notes, and missing timing.
