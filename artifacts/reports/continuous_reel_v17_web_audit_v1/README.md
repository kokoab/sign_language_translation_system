# Locked100 continuous-ASL metadata audit

Small public metadata files were acquired; no videos were downloaded and no examples were admitted to training.

## NCSLGR

[Official download documentation](https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html) links the index acquired as `video_index.zip` (564,370 bytes). Its 5,219 rows include multiple views of 1,887 utterances from 38 XML collections. Excluding the existing ncslgr10a–d subset leaves 1,721 utterances.

Case-sensitive exact Citizen raw-gloss matching found **77 textual contiguous-span candidates across 76 parent utterances, 42 distinct sequences and 39 raw classes** outside that subset. This is candidate discovery only: spelling equality is not lexical-variant approval, and the CSV has no per-sign timing or participant identity for safe crops/splits. No aliases or numeric variants were merged. Quoted classifier/gesture descriptions are kept as one token, and unsupported signs break a run.

See `ncslgr_phrase_candidates.json`, `ncslgr_index_audit.json`, and runnable `audit_index.py` (includes boundary/tokenization assertions). The [current DAI](https://dai.cs.rutgers.edu/dai/s/daioriginal) explicitly requires login for downloading search results, including the XML needed next. The official page also warns about half-speed compressed versions. Preserve existing signer roles and verify timebase before acquiring candidate clips.

## MoLo

[Gallaudet’s collection page](https://ida.gallaudet.edu/molo/) links OSF transcripts. The public API succeeded despite earlier browser access failures. Complete listing: **32 files, of which 16 are ELAN `.eaf` transcripts**; the other files are preferences. This is not 46 fully annotated participant recordings.

Downloaded and parsed one transcript, `MoLo002_N_KennethDeHaan_DHC_Acts_220426.eaf`, via its publisher-linked public download URL. It contains 299 right-hand and 124 left-hand ID-gloss annotations with explicit millisecond alignments.

In that sample, exact locked-raw-label matching found 34 right-hand and 13 left-hand occurrences across 13 distinct raw labels. These counts include possible two-hand duplicates; they are not distinct signs or ready phrases.

See `molo_audit.json` for the 16 transcript download links and timed sample matches. Two-hand alignment, uncertain annotations, exact variants and source-video correspondence must be audited before creating contiguous training targets. The publisher states videos are CC BY-NC-SA; transcripts remain under development. No full-corpus coverage is claimed.

## Next usable action

Use the 77 NCSLGR candidate spans to request/download the relevant annotated utterances through the authorized DAI account, and audit the listed MoLo transcripts for aligned target-only spans with participant identities. Deduplicate against existing collections, reserve signers before training, and keep every synchronized view of a performance together. Full remote training data acquisition remains gated by these concrete annotation checks.

The delegated audit failed with a usage limit and produced no artifact; the main agent completed this bounded metadata audit. Sources and hashes are saved alongside the report.
