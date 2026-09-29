# ATLAS final manuscript review

Reviewed: 2026-09-30. Scope: revised Chapters 1–4, navigation, citations, comparison tables and linked figures. The original manuscript is unchanged. No Google Docs content was modified during this review.

## Corrections made

- Repaired contents links and removed obsolete page numbers from the Markdown navigation. Standardized chapter/subsection headings and figure/table captions.
- Corrected the MediaPipe Hands citation to Zhang et al. (2020), added the missing Jiang et al. (2021) reference and distinguished the two Renjith author groups in citations.
- Clarified that the architecture/family comparisons concern landmark-based classifiers, rather than the complete multimodal deployment. Defined abbreviated model families and parameter units.
- Corrected the iPhone timing cross-reference and identified all FP16 visual/recognition components consistently. Retained the separation between conversion agreement, recognition accuracy, development-computer timing, model storage and the combined 28 ms iPhone measurement.
- Narrowed the English-output metric to retention of NO, matching the recorded assessment rather than claiming coverage of all ASL negation.
- Replaced the Activity List with the table from the supplied Google Docs export, as requested. This restores the source task identifiers and durations, including 30 days for mobile application development. Reworded the schedule discussion to describe planning without claiming that every milestone was completed.

## Points still requiring discussion

1. **Chapter 2 subset attribution.** The original passage remains unchanged at the author's request. The cited ASL Citizen paper does not establish an official standardized 100-sign subset or the attributed four-step selection procedure. This is an unresolved factual attribution, not a verified claim. The related categorical comparison with other datasets also needs source support. The supported alternative is to describe ATLAS's selected vocabulary and document its actual selection method. [ASL Citizen paper](https://arxiv.org/html/2304.05934v2).
2. **Source schedule dependencies.** The Google Docs schedule is now the authority, but its dependency notation needs clarification. AR (interface work) lists AP (mobile development) as a predecessor although both start May 12 and AR ends before AP. This can represent planned overlap, but it is not a finish-to-start dependency. AM spans April 7–May 11 and is labelled ethics-clearance approval while the review activities occur within that interval. Confirm whether AM is an umbrella activity or an approval milestone before interpreting the PERT critical path. Source dates and diagrams were retained without inventing corrections.
3. **Campus language context.** The scope correctly identifies ASL and does not claim Filipino Sign Language coverage. If the paper claims suitability for particular campus signers, that claim needs evidence about their language use; the current wording presents the campus as an intended context.

## Google Docs transfer checks

- Embed the local images rather than copying their relative filesystem paths. Rebuild the contents and page numbers using the final Google Docs layout.
- The results introduction links to `ATLAS_MEASUREMENT_NOTES.md`. A local Markdown link will not work for Google Docs readers: provide an accessible companion document or transfer its relevant provenance into an appendix before submission.
- Preserve the distinction between planned quality-assessment criteria and measured results. Do not add participant ratings, signer counts or independent-test claims unsupported by the evaluation records.

## Verification

The manuscript retains four objectives, 21 sequential figure captions and 19 sequential table captions. All embedded image paths and contents anchors resolve locally. Conversion values were retained; no training or evaluation was rerun. The original manuscript SHA-256 remains `42b36f7518152627d2e47f9d438fa058843ed0e3763527e26290c1559a159b25`.

Citation correction source: [Jiang et al., CVPR Workshops 2021](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeleton_Aware_Multi-Modal_Sign_Language_Recognition_CVPRW_2021_paper.html). MobileCLIP2 attribution was checked against [Apple's research page](https://machinelearning.apple.com/research/mobileclip2).
