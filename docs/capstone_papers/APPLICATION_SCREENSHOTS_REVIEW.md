# Application screenshots — manuscript review

Status: applied to the Google Docs Manuscript tab (`t.d1buvx4fb1j1`), Section 4.3.3, on 13 October 2026. After the 14 October condensation, ten images are arranged as Figures 10–14: startup/home, Live, Practice, Glosses, and paired History/session detail. The original wording below remains as the preparation record. Portrait images are 90–100 pt wide; Live images are 200 pt wide, paired side by side, and Practice landscape is 260 pt wide. Captions and descriptions remain with compact figures. Source and editing notes below are retained for audit, not inserted as manuscript text.

## Placement and scope

- [x] Insert the walkthrough in **4.3.3 Application Features and Operation**, under **4.3 Description of the Prototype**.

Replace the current brief feature description and screenshot placeholder with the text below. Preserve 4.3.1 Hardware Setup and Operating Environment and 4.3.2 Final Models and Processing Components. Preserve the SDLC, PERT, Gantt chart and other system diagrams. Keep measured performance and testing findings in 4.5 Implementation Results. The existing HELLO–HOW–YOU processing illustration remains separate.

Use unnumbered screen headings within 4.3.3. Figure letters A–F below are review placeholders; assign sequential manuscript figure numbers when inserting the approved images and update the List of Figures and cross-references.

## Image preparation notes — not manuscript text

- [ ] Capture the actual current application on the **iPhone 13**. Use the installed interface, including its current toolbar; use landscape for Live and Practice and portrait for the remaining screens; do not restore removed Settings or Letters controls for illustrations.
- [ ] Use real past session material for the signer images. Record the source session and frame timestamp for every reused frame. Keep reference demonstration clips distinct from the user's session footage.
- [ ] Live and Practice use captured iPhone 13 layouts with signer imagery inserted from the supplied session video. Keep source/editing details in this review file, not as text burned into the images. These assembled interface views are not unmodified screenshots of a new phone recognition run.
- [ ] Preserve actual recognition and English outputs when presenting a recorded session. Do not invent successful outputs, scores, timestamps or saved records. An illustrative layout does not establish recognition performance.
- [ ] Keep a source register with image filename, device/build, capture date, session/frame source and any compositing or cropping. Remove incidental personal information without altering the demonstrated result.
- [ ] Keep captions with figures, make interface text readable at manuscript size, and use paired panels only where both remain legible. Capture missing states before finalizing their descriptions.

The iPhone 13 is the capture device; the feature discussion below remains platform-neutral. These notes are editorial instructions, not additional manuscript paragraphs.

## Proposed manuscript wording and figures

### 4.3.3 Application Features and Operation

- [x] **Opening paragraph**

The mobile application provides Live, Practice, Glosses and History functions. These functions allow users to capture signing, inspect recognized signs and generated English, view vocabulary demonstrations, practise supported signs and review previous sessions. The following figures present the application screens in the order in which users encounter these functions.

### Application startup and home screen

- [x] **Figure A and discussion**

**Figure A. ATLAS splash screen and home screen.**

| (a) Splash screen | (b) Home screen |
|---|---|
| ![Splash](review_assets/application_screens/splash.png) | ![Home](review_assets/application_screens/home.png) |

The splash screen presents the ATLAS name and application identity during startup. The home screen provides access to live translation through Start translating and to the Practice, Glosses and History functions. It also provides access to the application guide. These entry points organize the application's main functions in a common navigation screen.

**Review note:** Pair these screens because they form one short startup sequence. The actual splash includes a startup progress indicator. Do not borrow the DIRTX backend-connection explanation.

### Live sign recognition

- [x] **Figure B and discussion**

**Figure B. Live sign recognition interface.**

**(a) Ready state**

![Live ready screen](review_assets/application_screens/live_landscape.png)

**(b) Active recognition**

![Live recognizing HELLO](review_assets/application_screens/live_recognized_landscape.png)

**Review source note:** Landscape controls were captured on the iPhone 13. The camera images come from the supplied `20260930_134459_04.mp4`; the active view uses its 0.8-second frame. The current native mobile recognition code and bundled FP16 models were replayed on the Mac. The measured output was HELLO, GOOD, MORNING, HOW, YOU. HELLO scored 0.99329798 (displayed as 99%); its recorded identity-commit time was 0.78 seconds. The active view is assembled from that evidence, not a screenshot of the replay running on the phone. The last YOU was emitted during the end-of-file flush. No English output is invented.

The Live interface displays the device camera preview together with recognition feedback, accepted glosses and generated English. Camera frames provide landmark and hand-image inputs for on-device inference. The boundary detector and recognizer support the identification of candidate sign intervals, while the decoder determines which glosses are accepted into the sequence. The visible output allows users to inspect the recognized message during the interaction. Camera and landmark-display controls support the capture and viewing of signing input.

**Review note:** This figure shows the actual user interface. Keep the detailed model architecture in 4.3.2 and avoid repeating its full technical description here.

### Message completion and output

- [x] **Figure C and discussion**

**Figure C. Recorded signs and retained English output from a saved session.**

![Actual saved session output](review_assets/application_screens/session_detail.png)

**Capture note:** Actual saved-session detail from 13 October 2026, 20:37. This shows the retained output, not the Live Finish screen or a new accuracy measurement. The displayed English is preserved as recorded; it is not asserted to translate every gloss in the session log.

The Finish control indicates that the user has completed the message. The application finalizes the remaining text and provides speech output for the completed English message. Displaying the accepted glosses and generated English allows the user to inspect both the recognized sign sequence and its English rendering. Figure C shows how the application retains the recognized signs and English output for subsequent review.

**Review note:** Select the example after examining past sessions; no particular sentence or successful result is asserted in advance. Speech is described as functionality, not something proven by a still image. Figures B and C may share a paired figure if the text remains readable.

### Practice

- [x] **Figure D and discussion**

**Figure D. Practice selection and signing interface.**

**(a) Practice-set selection**

<img src="review_assets/application_screens/practice_selection.png" alt="Practice selection" width="340">

**(b) Landscape practice interface**

![Landscape practice](review_assets/application_screens/practice_landscape.png)

**Review source note:** The Practice layout and its FRIEND reference thumbnail were captured on the iPhone 13. The camera background was replaced with the supplied session video frame at 2.0 seconds. This shows the interface awaiting an attempt; it does not claim a successful FRIEND recognition or a completed practice score.

The Practice function allows users to select a set of supported signs and attempt them using the device camera. Reference demonstrations provide a visual example of the requested sign, while the interface presents feedback during the practice activity. Users can proceed through the selected set and review the practice summary afterward. The feedback reflects the recognizer's response to the captured input and should not be interpreted as an independent assessment of signing proficiency.

**Review note:** Keep the built-in reference demonstration authentic. The recorded signer background is an interface illustration, not a newly completed practice attempt.

### Vocabulary browsing

- [x] **Figure E and discussion**

**Figure E. Glosses vocabulary browser and sign demonstration.**

| (a) Local video search results | (b) Local sign demonstration |
|---|---|
| ![Local Glosses entries](review_assets/application_screens/glosses_local.png) | ![Local ANSWER demonstration](review_assets/application_screens/gloss_detail_local.png) |

**Capture note:** The visible browser results are ANGRY and LANGUAGE, obtained with the actual search term ANG. Both entries, and the ANSWER detail example, are marked `source: local` in the application catalog. No other-corpus thumbnails are shown. This is a search selection, not a source-filter feature.

The Glosses function provides access to the application's supported vocabulary. Users can browse or search for a sign and open its detail screen to view a demonstration. Playback controls support closer examination of the example before attempting the sign. This function makes the available vocabulary visible and provides a reference for interaction with the recognition component.

**Review note:** Use an existing supported entry and its actual demonstration. The figure documents vocabulary access; it does not establish coverage beyond the selected vocabulary.

### Session history

- [x] **Figure F and discussion**

**Figure F. Session history.**

![Actual History list](review_assets/application_screens/history.png)

The corresponding saved-session detail is shown in Figure C; do not duplicate that screenshot here.

The History function lists previous sessions for review. Opening a session displays its recorded signs and available English output, together with a sign timeline. The detail screen also identifies sessions that ended before Finish. These records allow users to revisit earlier interactions and inspect the output retained by the application.

**Review note:** Use actual saved records. Prefer the same session used for the completed-message illustration when its corresponding History entry is available, so the walkthrough forms one traceable example.

- [x] **Closing paragraph**

Together, these screens connect sign capture, recognition feedback, English output, vocabulary reference and session review within the mobile application. Recognition remains limited to the supported vocabulary and depends on the quality and completeness of the captured input. Quantitative performance and software-testing results are reported in Section 4.5.

## Figure source register

| Figure | Image files | Source and preparation |
|---|---|---|
| A | splash.png, home.png | Actual iPhone 13 startup capture and mirrored home capture |
| B | live_landscape.png, live_recognized_landscape.png | Landscape iPhone 13 layout; supplied video at 2.0 s and 0.8 s; active output from fresh native-code Mac replay |
| C | session_detail.png | Actual mirrored saved-session detail, 13 October 2026 at 20:37; outputs unchanged |
| D | practice_selection.png, practice_landscape.png | Actual mirrored selection; landscape iPhone 13 Practice layout with recorded signer background; original local reference preserved |
| E | glosses_local.png, gloss_detail_local.png | Actual mirrored app views; only local ANGRY/LANGUAGE search results and local ANSWER demonstration |
| F | history.png | Actual mirrored History list; no records invented or edited |

Images are stored under `review_assets/application_screens/`. The interface illustrations were prepared with image generation; original captures and replay evidence are retained separately. No disclaimer text is burned into the finished images.

### Replay evidence — review only

- Supplied source: `/Users/frnzlo/Downloads/20260930_134459_04.mp4`, duration 5.121667 s, 1280 × 720.
- SHA-256: `f79c693908a1ed16ed6da5f1aeb4a71efe8bf0b1fe31a4158984f19ee5cadb53`.
- This supplied copy differs from the longer same-named file pulled from the phone. The supplied copy is the source used for the final illustrations and replay.
- Current native Core, Models, Decoder and Engine sources compiled into an isolated Mac replay harness; no production code or model changes.
- Result: `artifacts/reports/paper_app_screens_20261013/replay_result.json`.
- This demonstration does not replace the paper's aggregate evaluation or phone latency measurements. The HELLO percentage is a model score for this example, not accuracy.

## Basis for the proposed placement

DIRTX_MANUSCRIPT.pdf places its splash screen and subsequent interface walkthrough under Description of the Prototype, beginning on printed page 42. Its descriptions explain the screen, available actions and resulting system behavior. ATLAS follows that organizational approach while retaining its own implemented features and current Chapter 4 structure.

Verified drafting references: CHAPTER_4_TEMPLATE_ALIGNED_DRAFT.md, PROJECT_GROUND_TRUTH.md (2026-10-13 mobile interface entries), and the current application shell sources in ATLAS/mobile_app/slt_mobile_app/lib/shell/. The approved images and prose are now inserted. The source notes remain in this local review file.

## Application record

- Target: Manuscript tab, Section 4.3.3 Application Features and Operation.
- Figure mapping: A→10, B→11, C→12, D→13, E→14, F→15. Former Figures 10–15 are now 16–21; matching cross-references and list entries were updated.
- New figures appear on printed pages 88–93. Captions use left/right or top/bottom labels matching the rendered arrangement.
- Preserved the existing SDLC, PERT, GANTT, model/results content and all other document tabs.
- Render evidence: `artifacts/reports/paper_app_screens_20261013/`.
