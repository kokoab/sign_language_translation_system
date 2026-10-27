You are editing a Google Doc for me. Use the Google Docs connector to edit the document in place (same file, same link). Do not create a new document.

Document: https://docs.google.com/document/d/1i-sVr5ny6vmf6QDb_Dbr1e_CA3_xH7y7s68-wVqaSzc/edit
Edit ONLY the tab named "Revised Manuscript". Do not touch the "Post-Defense Revision Plan" or "Manuscript" tabs.

Rules:
1. Apply the changes below exactly as written. Do not reword, shorten, or "improve" the text. Copy it character for character, including numbers and punctuation.
2. "Replace the paragraph starting '…'" means: find the paragraph in the Revised Manuscript tab that begins with those words and replace the whole paragraph with the new text. If you cannot find a unique match, stop and ask me instead of guessing.
3. Match the surrounding formatting: body paragraphs like the neighbouring body text, section headings like the existing headings at the same level (e.g. "4.8.2 …"), table and figure captions like the existing captions ("Table 17." bold, title italic).
4. Insert tables as real Google Docs tables (header row first), not as text with "|" characters.
5. Figures: I will insert the two images myself. Where a figure goes, insert a placeholder paragraph exactly "[INSERT IMAGE: pipeline_flowchart.png]" or "[INSERT IMAGE: calibration_reliability.png]" above its caption.
6. Work section by section, top to bottom. Read the document again before each edit so positions are current. After all edits, re-read the tab and check every change landed once and in the right place.
7. Then renumber so the tab is consistent. Apply this map to ALL existing captions AND every in-text reference ("Figure 12", "Table 15", "Section 4.8.4", etc.) in the Revised Manuscript tab. The new text below already uses the final numbers, so do not renumber it again:
   - Figures: new pipeline flowchart = Figure 9; existing Figures 9–17 become 10–18; new reliability diagram = Figure 19.
   - Tables: Tables 1–10 unchanged; new metrics table = Table 11; existing Tables 11–16 become 12–17; new phone-results table = Table 18; existing Table 17 (English-Output Comparison) becomes Table 19; existing Table 18 is removed and replaced by the new Table 20; existing Tables 19–26 become 21–28.
   - Sections: new 4.8.3; existing 4.8.3 (English) becomes 4.8.4; existing 4.8.4 becomes 4.8.5; new 4.8.6; existing 4.8.5 and 4.8.6 become 4.8.7 and 4.8.8. Update the Table of Contents, List of Figures and List of Tables to match.
8. When done, give me a short list: each change, where it landed, and anything you could not do.

The changes, in document order:


=== CHANGE 1 (P-17) — Section 1.3, Objective 2 ===

**Replace** Objective 2 with:

> Compare the performance of the implemented recognition models or configurations using Top-1 accuracy and parameter count for isolated-sign classifiers, Top-5 accuracy, macro-averaged precision, recall, and F1, and expected calibration error (ECE) where recorded; Word Error Rate (WER), sequence precision, sequence recall, and commit latency for streaming recognition; frame processing time for FP32 and FP16 model configurations on the target device; and the Bilingual Evaluation Understudy (BLEU) score for English-generation modes, to identify the configuration that best meets the stated selection criteria.

=== CHANGE 2 (P-01) — Section 3.2 ===

**Replace** the paragraph starting "Finally, English Generation and Speech Output…" with:

> Finally, English Generation and Speech Output uses a fine-tuned T5-efficient-tiny model to generate English from the accumulated glosses. English preparation takes place during signing, with completed sentences retained while the remaining input is processed. The Finish button, or a gesture in which the signer holds the right hand open and the left hand in a fist above the shoulders for one second, signals completion of the message. The system then finalizes the remaining English text and provides local speech output. On-device inference executes the exported models locally, connecting the trained components with the device camera, display, and completion controls.

=== CHANGE 3 (P-02) — Section 4.5, "Flowchart" subsection ===

**Keep** the existing navigation paragraph and rename its figure to "Figure 8. *Application Navigation Flowchart*". **Add after it:**

> Figure 9 shows the processing sequence that converts live signing into English text and speech. The device camera supplies frames at 30 frames per second. Because each frame takes several tens of milliseconds to analyze, the application processes at most one frame every 50 milliseconds (20 frames per second) and always takes the newest available frame, so recognition does not fall behind the signer. For each processed frame, landmark extraction locates 61 points on the hands, face, and upper body. The boundary model uses these landmarks to estimate, for every frame, whether the signer is outside a sign, beginning one, or inside one; it looks 0.3 seconds ahead before deciding. Meanwhile, crops of the left hand, right hand, and both hands are encoded by MobileCLIP2-S0 to describe handshape appearance. Candidate sign intervals proposed by the boundary model are resampled to 32 frames and scored by the Squeezeformer-based recognizer, which combines the landmark and hand-image evidence to produce scores for the 100 supported signs. The segmental decoder combines these sign scores with the boundary estimates and decides whether a sign is complete and stable enough to commit. A committed sign is appended to the visible gloss buffer, and the English text is updated from the accumulated glosses. When the user taps Finish or holds the Finish gesture for one second, the open sign is flushed, the English text is finalized, and the device reads it aloud. Until then, the sequence repeats for the next frame.
>
> **Figure 9.** *ATLAS Processing Pipeline Flowchart*

Put "[INSERT IMAGE: pipeline_flowchart.png]" directly above the "Figure 9." caption.

=== CHANGE 4 (P-04) — New subsection at the end of "Data Preparation and Model Development", just before Section 4.6 ===

> **Evaluation Metrics and Parameter Selection**
>
> Accuracy, precision, and recall computed on individual clips describe a classifier that receives a sign that has already been cut out of a recording. ATLAS does not receive such clips during use. It observes a continuous camera stream and must decide where each sign begins and ends, which sign it is, and when the decision is stable enough to show. An error in any of these decisions changes the output even when the classifier itself is accurate. Each component was therefore evaluated with the metric that reflects its own task, and the complete pipeline was evaluated on continuous recordings with sequence-level and time-based metrics (Zuo et al., 2024).
>
> The same metrics were used to choose each component's settings, as summarized in Table 11. For example, the boundary model's look-ahead was chosen by weighing WER against commit latency, and the use of FP16 models was chosen by frame processing time, which in turn affects WER on the phone. The results in Section 4.8 therefore report the same measures that guided these choices.
>
> **Table 11.** *Pipeline Components, Evaluation Metrics, and the Settings They Determined*
>
> | Component | Metric | Setting determined by the metric |
> |---|---|---|
> | Landmark extraction | Extraction time per frame | Apple Vision: 10.7 ms per frame, compared with 73.1 ms for MediaPipe Holistic under the same load on the development computer. |
> | Landmark representation | Top-1 accuracy on rotated inputs | Camera-roll augmentation of ±12°, which kept accuracy at 94.87–95.32% across sampled rotations. |
> | Hand-image features | Top-1 accuracy | Hand images included: 95.40% combined, compared with 95.27% for landmarks alone. |
> | Sign recognizer | Top-1 accuracy; macro precision, recall, F1; development WER | Part-wise and global Squeezeformer (96.83% in the base comparison); checkpoint at epoch 4, the first with the lowest development WER while retaining isolated accuracy. |
> | Boundary model | WER and commit latency | Look-ahead of 0.3 s: longer look-ahead lowered WER but delayed each sign, and the design target was for signs to appear within 0.5 s of their end. |
> | Segmental decoder | Sequence precision, recall, WER | Commit threshold of 0.4, which reduced WER and extra outputs compared with 0.5. |
> | Complete pipeline | WER; commit latency; frame processing time | 20 frames per second with FP16 models, which kept frame time within the 50 ms available per frame (Section 4.8.5). |
> | English generation | Judged correctness; BLEU | Incremental generation, which matched whole-sequence correctness while preparing sentences during signing (Section 4.8.4). |

=== CHANGE 5 (P-05) — New paragraph directly after the new Table 11 from the previous step ===

> Word error rate (WER) aligns the recognized gloss sequence with the reference sequence and counts substitutions (S), deletions (D), and insertions (I): WER = (S + D + I) / N, where N is the number of reference signs. Sequence precision is the share of output signs that are correct, and sequence recall is the share of reference signs that are recognized; F1 is their harmonic mean. Unlike clip accuracy, these values penalize missed boundaries, repeated outputs, and signs inserted during transitions. Commit latency is the time from the end of a sign, as estimated by the boundary model, to the moment its gloss is available, including the processing time of the frame that commits it. Frame processing time is the time the application takes to analyze one camera frame; the processed frame rate and the real-time factor (total processing time divided by recording duration) show whether the pipeline keeps pace with the camera. Expected calibration error (ECE) measures whether the recognizer's confidence matches its accuracy: predictions are grouped by confidence into 15 equal-width bins, and the absolute difference between average confidence and accuracy in each bin is averaged, weighted by the number of predictions in the bin (Naeini et al., 2015; Guo et al., 2017).

=== CHANGE 6 (P-08) — Section 4.7.2 ===

**Replace** the paragraph starting "The selected models are exported for on-device inference…" with:

> The selected models are exported for on-device inference. Visual and recognition components use FP16 (16-bit floating point); English generation retains FP32 (32-bit floating point). Table 12 identifies the deployed components. To measure the effect of numerical precision, the hand-image encoder, boundary model, and recognizer were also exported in FP32 from the same trained checkpoints, using the same export procedure with only the precision setting changed. The two configurations were compared on recognition accuracy using the same validation inputs, on model package size, and on iPhone 13 processing time and WER using the same recordings (Section 4.8.5).

=== CHANGE 7 (P-09) — New paragraph directly after the paragraph from the previous step ===

> Processing time was measured on an iPhone 13 (iOS Version 26.7.2 (Build 23H127)) running a release build of ATLAS. Recordings from the evaluation set were supplied to the application's frame-processing path at their original timing, so frames were selected and skipped exactly as with live camera input. The processing time of each frame was measured with the device clock from the moment the engine received the frame until it returned its result. This covers landmark extraction, hand-image encoding, boundary estimation, recognition, and decoding, but not camera capture, model loading, interface drawing, English generation, or speech. Each configuration was run twice in alternating order (FP16, FP32, FP32, FP16) so that the warming of the device affected both equally, and the median and 95th percentile over all processed frames are reported. The resulting millisecond value is the time needed to analyze one camera frame, not the delay before a sign appears, which is reported separately as commit latency.

=== CHANGE 8 (P-11) — Section 4.8, opening paragraphs ===

**Replace** the two paragraphs starting "Results address recognition…" and "Each comparison retains…" with:

> Results address recognition, connected signing, English generation, device execution, calibration, and software testing, using the metrics defined in Table 11 and the accompanying definitions. Top-1 accuracy measures correct highest-scoring predictions on isolated clips; WER, sequence precision, and sequence recall measure the complete pipeline on continuous recordings; commit latency and frame processing time measure response time; and ECE measures confidence calibration.
>
> Each comparison retains its task and measurement conditions. Base-classifier results describe the original validation study; adapted recognition components share one validation dataset. Sequence results first track five stages of system development, with the change from the training-experiment procedure to runtime evaluation identified explicitly, and then report the final configuration on iPhone 13 using the evaluation set.

=== CHANGE 9 (P-19) — Section 4.8.2 ===

**Replace** the paragraph starting "Stages 1–2 share the training-experiment evaluation procedure…" with:

> Stages 1–2 share the training-experiment evaluation procedure; stages 3–5 use runtime boundary detection, interval selection and gloss acceptance, evaluated with the Python implementation used during development. The recordings supported development and model selection, not independent final testing. Differences across these procedures cannot be attributed to recognizer training alone.

**Replace** the paragraph starting "Refinements to ordering and acceptance reached 12.83% WER…" with:

> Refinements to ordering and acceptance reached 12.83% WER in this development implementation; subsequent transition-example fine-tuning retained that result. Improvement was not uniform: an intermediate fragment/rejection variant recorded 15.93%. These stages describe how the configuration was developed; the performance of the final configuration on the target device is reported in Section 4.8.3.

**Rename** the figure caption to: *Figure 18. Connected-Sign WER Across Development Stages (Development Implementation)*

=== CHANGE 10 (P-13) — New Section 4.8.3, inserted after Section 4.8.2 (after Figure 18) and before the English-generation section ===

> The final configuration was evaluated on iPhone 13 using the evaluation set of continuous phrase recordings from Section 4.8.2. The recordings were supplied to the application at their original timing, so the application analyzed and skipped frames exactly as it does with a live camera (Section 4.7.2). Each recording was processed twice, and Table 18 reports the pooled results.
>
> **Table 18.** *Streaming Performance on iPhone 13 (FP16 Configuration)*
>
> | Measure | Value |
> |---|---:|
> | WER | 15.93% |
> | Sequence precision | 91.9% |
> | Sequence recall | 88.1% |
> | F1 | 89.9% |
> | Commit latency, median | 0.26 s |
> | Commit latency, 90th percentile | 0.68 s |
> | Signs available within 0.5 s | 69.0% |
> | Frame processing time, median | 36.92 ms |
> | Frame processing time, 95th percentile | 61.57 ms |
> | Frames analyzed per second | 19.9 |
>
> The system recorded 15.93% WER, with 91.9% sequence precision and 88.1% sequence recall. The median commit latency was 0.26 s, and 69.0% of signs were available within 0.5 seconds of their estimated end. The median frame processing time was 36.92 ms (95th percentile 61.57 ms), and the application analyzed 19.9 frames per second, close to its limit of 20. During this continuous evaluation, the device's thermal state reached "serious", so these times reflect sustained use rather than a cool device. English generation with T5-efficient-tiny took a median of 154 ms per utterance in total, including the sentences prepared during signing.

=== CHANGE 11 (P-21) — New paragraph directly after Table 18 in the new Section 4.8.3 ===

> The WER of the final configuration on the phone was higher than the 12.83% reported in Section 4.8.2, for two reasons. First, the development results in Section 4.8.2 were obtained with the Python implementation used during development, which runs the trained models in full precision on cached inputs. The application uses a separate Swift implementation with FP16 Core ML models. When the same recordings were replayed with the application's implementation on the development computer, in the same way as in Section 4.8.2, WER was 15.04%; the difference from 12.83% therefore arises between the two implementations, and its exact source was not isolated. Second, on the phone, frames that arrive while the previous frame is still being analyzed are skipped, as with a live camera, which raised WER to 15.93% (two passes: 15.04% and 16.81%). This effect was much larger for the slower FP32 configuration, which missed more than twice as many signs (Section 4.8.5).

=== CHANGE 12 (P-14) — Current Section 4.8.4 (becomes 4.8.5) ===

**Replace** all of current Section 4.8.4 (from "The FP16 recognizer export was evaluated…" through the 31.14 ms paragraph, including Table 18) with:

> Table 20 compares the FP32 and FP16 configurations of the visual and recognition models. Both were exported from the same trained checkpoints; only numerical precision differs. English generation used the same FP32 model in both configurations.
>
> **Table 20.** *FP32 and FP16 Configurations*
>
> | Measure | FP32 | FP16 |
> |---|---:|---:|
> | Primary recognizer validation accuracy | 95.50% | 95.50% |
> | Recognizer package | 52.38 MB | 26.56 MB |
> | Boundary-model package | 9.29 MB | 4.68 MB |
> | Hand-image encoder package | 45.58 MB | 22.89 MB |
> | iPhone 13 frame processing time, median | 66.06 ms | 36.92 ms |
> | iPhone 13 frame processing time, 95th percentile | 116.76 ms | 61.57 ms |
> | Frames analyzed per second | 14.1 | 19.9 |
> | WER | 23.45% | 15.93% |
>
> Both precisions produced the same prediction for 3,957/3,957 validation clips, so numerical precision did not change recognition. On the phone, however, FP32 needed a median of 66.06 ms per frame, more than the 50 ms available at 20 frames per second, so it analyzed only 14.1 frames per second and its WER was 23.45%, compared with 15.93% for FP16. FP16 also halved the size of each package, and it was therefore retained for the visual and recognition models. Package sizes are complete Core ML packages in decimal megabytes, not working memory or total application size.

=== CHANGE 13 (P-15) — New Section 4.8.6, inserted after the section from the previous step ===

> Posterior calibration describes whether the confidence that the recognizer assigns to a prediction matches how often such predictions are correct; for example, among predictions made with 90% confidence, about 90% should be correct. Calibration was measured on the isolated-sign validation clips using expected calibration error (ECE), which groups predictions by confidence and averages the gap between each group's confidence and its accuracy; an ECE of 0% means that confidence and accuracy agree exactly. Before any correction, the recognizer's average confidence was 92.60% while its accuracy was 95.50%, so it was slightly underconfident, and its ECE was 3.03%. Temperature scaling divides the recognizer's output scores by a single constant before they are converted into probabilities (Guo et al., 2017). With a constant of about 0.84, fitted on one half of the clips and checked on the other half, ECE fell to 1.02%. Temperature scaling changes only the confidence values, not which sign is ranked first, so accuracy was unchanged. The correction improved the ASL Citizen and locally recorded clips but slightly worsened the SemLex clips. Because the decoder's commit threshold was chosen using the original scores, the deployed application keeps those scores, and calibration is reported here as an evaluation of the recognizer's confidence.
>
> Figure 19 is a reliability diagram of these results. Each point represents a group of predictions with similar confidence: its horizontal position is the group's average confidence, and its vertical position is the share of the group that was correct. The dashed diagonal marks perfect calibration. Points above the diagonal show underconfidence, where the recognizer was right more often than it claimed, and points below it show overconfidence. The blue line, before correction, lies mostly above the diagonal at high confidence. The orange line, after temperature scaling, lies on the diagonal in the highest-confidence group, which contains most of the predictions. Groups at lower confidence contain few clips, so their points vary more; groups with fewer than ten clips are not drawn.
>
> **Figure 19.** *Reliability Diagram of the Primary Recognizer*

Put "[INSERT IMAGE: calibration_reliability.png]" directly above the "Figure 19." caption.

=== CHANGE 14 (P-16) — Section 5.1 ===

**Replace** the paragraph starting "The results showed that the primary recognizer achieved 95.50%…" with:

> The results showed that the primary recognizer achieved 95.50% isolated-sign validation accuracy in both FP32 and FP16, and its expected calibration error was 3.03%. On iPhone 13, the complete streaming pipeline recorded 15.93% WER, with 91.9% sequence precision and 88.1% sequence recall, a median frame processing time of 36.92 ms, and a median of 0.26 s from the end of a sign to its display. However, these results were obtained from development and model-selection recordings rather than an independent final test set. For English generation, both whole-sequence and incremental generation achieved a 60% fully correct rate under the automated evaluation procedure, indicating that further improvements are needed to produce more consistent and accurate English output.

=== CHANGE 15 (P-20) — Section 5.2 ===

**Replace** the paragraph starting "Furthermore, broader performance testing should be conducted…" with:

> Furthermore, broader performance testing should measure the complete delay from camera capture to spoken output. The present evaluation timed frame analysis, commit latency, and English generation separately on one device; camera capture and speech output were not timed, and the device reached a high thermal state during continuous use. The difference between the development implementation and the application's implementation should also be isolated, so that improvements measured during development carry over to the application. The application should also be evaluated across supported mobile devices and over longer sessions to assess its performance, stability, and resource usage under different operating conditions.

=== CHANGE 16 (P-18) — References list ===

> Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning* (Vol. 70, pp. 1321–1330). PMLR. https://proceedings.mlr.press/v70/guo17a.html
>
> Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. *Proceedings of the AAAI Conference on Artificial Intelligence, 29*(1). https://doi.org/10.1609/aaai.v29i1.9602
>
> Zuo, R., Wei, F., & Mak, B. (2024). Towards online continuous sign language recognition and translation. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing* (pp. 11050–11067). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.619

Add each reference to the References list in alphabetical order by first author, in the same style as the existing entries.
