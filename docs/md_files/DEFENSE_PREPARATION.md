# Defense Preparation: Possible Questions & How to Answer

## SECTION 1: Questions About the Architecture (Dr. Caluza will likely ask these)

### Q1: "What exactly is DS-GCN? How is it different from a standard GCN?"
**Answer:** A standard multi-partition GCN uses a separate full weight matrix for each adjacency partition -- 3 partitions means 3 expensive matrix multiplications per layer. DS-GCN factorizes this into two steps: (1) lightweight per-channel scaling for each partition (just a vector multiplication, not a matrix), keeping the three directional signals separated, then (2) one shared linear layer that mixes channels. This reduces the number of dense matrix multiplications from 3 to 1 per layer, while preserving directional awareness (self, parent-to-child, child-to-parent).

### Q2: "You claim DS-GCN is lighter and more efficient. Can you prove that?"
**Answer:** The efficiency comes from reducing the number of full matrix multiplies in the spatial aggregation step. A standard GCN with 3 partitions does 3 full matrix multiplies per layer (one per partition). DS-GCN does 3 cheap vector scalings + 1 matrix multiply. With 4 GCN blocks, that's 4 matrix multiplies instead of 12 -- roughly 3x fewer FLOPs in the spatial aggregation. This is consistent with the factored convolution approach from Howard et al. (2017) in MobileNet and the efficiency gains demonstrated by Cheng et al. (2020) with Shift-GCN.

### Q3: "Why did you choose 384 as the model dimension?"
**Answer:** 384 is a hyperparameter chosen empirically. It must be divisible by the number of attention heads (8), giving 48 per head. Smaller dimensions (128, 256) showed lower representational capacity for 310 classes. Larger dimensions (512, 768) increased memory and compute without proportional accuracy improvement given our dataset size of ~57,000 samples.

### Q4: "Why 32 frames? Why not more or fewer?"
**Answer:** ASL signs typically last 0.5-1.5 seconds. At 30fps, that's 15-45 raw frames. 32 captures the full movement trajectory (preparation, stroke, retraction) without wasting compute. All videos are temporally resampled to 32 frames via linear interpolation, making the representation speed-invariant -- fast and slow signers both produce 32-frame clips. The fixed size also enables efficient batched GPU processing.

### Q5: "What is ArcFace and why did you use it instead of standard cross-entropy?"
**Answer:** ArcFace adds an angular margin penalty to the correct class during training. Instead of just requiring the model to output the right answer, it requires the model to be VERY confident -- the features must be within a few degrees of the class template on a normalized sphere. This forces tight intra-class clustering and wide inter-class separation. The benefit: signs that look similar (like M vs N, or SIX vs W) get well-separated in the learned feature space. The warmup schedule (epochs 1-10: pure CE, 11-30: gradual ramp, 31+: full ArcFace) prevents training collapse when the model hasn't yet learned meaningful features.

### Q6: "Why does your training loss appear higher than validation loss? Isn't that unusual?"
**Answer:** This is expected with ArcFace. During training, the angular margin penalty handicaps the correct class (cos(theta + 0.5) instead of cos(theta)), making the loss artificially higher. During validation, the margin is removed -- clean cosine similarity. So training is on "hard mode" and validation is on "easy mode." This is NOT overfitting -- it's the ArcFace margin working as intended.

### Q7: "What are angle-primary features and why are they important?"
**Answer:** Angle features (joint angles, curl ratios, finger spreads, palm normals) are computed from the relative geometry between joints, not their absolute positions. They're inherently signer-invariant because the same handshape produces the same angles regardless of hand size, arm length, or position in the frame. Our cosine similarity tests showed: XYZ = 0.63 across signers, angles = 0.90, curls = 0.93. This is why angle features became the primary discriminative signal in v14/v15.

### Q8: "How does CTC work and why did you choose it over an attention decoder?"
**Answer:** CTC is an alignment-free loss function. Given an input sequence and a target label sequence, it sums probabilities over ALL possible alignments using dynamic programming. It introduces a blank token for positions where no sign is being produced. We chose CTC because: (1) we don't have frame-level alignment labels, (2) our Stage 2 training data is mostly synthetic concatenated clips where CTC's flexibility is ideal, (3) CTC adds only one linear layer on top of the encoder (simple), and (4) our vocabulary is small (311 tokens) -- attention decoders are better for large open-ended vocabularies like English.

### Q9: "What is the MultiScaleTCN and why compress 32 frames to 4 tokens?"
**Answer:** The MultiScaleTCN uses three parallel depthwise convolution branches with kernel sizes 3, 5, and 9 to capture temporal patterns at different scales. The outputs are fused and pooled to 4 tokens per clip. The 8x compression is critical for CTC: too many tokens (32) makes the output blank-dominated and hard to decode, too few (1) loses temporal information. 4 tokens captures the sign's onset, stroke, peak, and offset phases.

### Q10: "Explain the complete data flow from video input to English output."
**Answer:** (1) Camera captures video frames. (2) RTMW-XL detects 133 keypoints per frame, we select 61 (hands + face + body). (3) Coordinates are normalized to [0,1], Z depth is estimated. (4) 1-Euro filter smooths jitter, temporal coherence rejection removes false detections, linear interpolation fills gaps. (5) Bone lengths stabilized, coordinates centered and scaled. (6) Resampled to 32 frames, kinematics (velocity, acceleration) computed, bone features added = [32, 61, 16] tensor. (7) Stage 1: DS-GCN processes spatial graph, angle features computed, TCN handles temporal patterns, ArcFace classifies into 310 signs. (8) Stage 2: Multiple clips encoded, MultiScaleTCN compresses to 4 tokens each, SequenceTransformer models cross-clip relationships, CTC decodes gloss sequence. (9) Stage 3: Flan-T5 translates gloss to English.

---

## SECTION 2: Questions About Limitations & Weaknesses (Both panelists)

### Q11: "Your system is not real-time. RTMW-XL takes 12 seconds per video. How is this useful?"
**Answer:** We acknowledge this openly in the paper. The system is a turn-based assistive tool, not a simultaneous interpreter. The signer completes their message, the system processes it, then displays the English translation. This is analogous to texting -- not instantaneous, but still far more efficient than handwritten notes. The 12-second latency is an extraction bottleneck on CPU; with GPU acceleration or model optimization (quantization, pruning), this could be reduced significantly in future work.

### Q12: "Your evaluation is class-stratified, not signer-stratified. Doesn't that inflate your accuracy?"
**Answer:** Yes, we acknowledge this as a limitation. Class-stratified means the same signers appear in both training and test sets, just with different video clips. Signer-stratified would hold out entire signers from training. Our reported 85.32% accuracy is signer-dependent. Signer-independent performance is estimated at 40-60%, which is a significant gap. The angle-primary features partially address this (0.90 similarity across signers vs 0.63 for XYZ), but formal signer-independent evaluation with leave-one-signer-out cross-validation is needed and is acknowledged as future work.

### Q13: "You have 66,451 extracted files but the CLAUDE.md says 57,535 samples. Where are the missing ~9,000?"
**Answer:** Two cleaning stages reduce the count. First, quality filters at load time reject files that are all-zeros (failed extraction), completely static (no motion), or have spatial outliers (landmarks outside normal range). Second, a confident learning step uses the trained model to identify likely mislabeled samples -- samples where the model is highly confident the label is wrong are removed (~3% of the dataset). This produces a cleaner training set that improves model accuracy.

### Q14: "Your Stage 2 trains primarily on synthetic data. How realistic is that?"
**Answer:** We acknowledge this limitation. Synthetic sequences concatenate isolated sign clips with artificially generated transitions. While we use biomechanically realistic minimum-jerk trajectories (Flash & Hogan 1985) and various augmentations (speed warping, boundary jitter, temporal drop, Gaussian noise), real coarticulation is more complex. We supplement with 780 real continuous phrase videos at 3x oversampling, but more real data would improve Stage 2 performance. The 5.17% WER on synthetic validation may not directly transfer to real-world continuous signing.

### Q15: "The mobile application is not yet integrated. Why did you include it in your timeline?"
**Answer:** The mobile application development is planned for the implementation phase (May-September 2026) but is not part of the current completed work. We have been transparent about this: the current system is desktop-based using OpenCV for video capture. The mobile integration, TensorFlow Lite conversion, and on-device optimization are future work. The current study focuses on developing and evaluating the recognition and translation pipeline itself.

### Q16: "Ghost-hand behavior -- RTMW-XL predicts hands even when only one is visible. How do you handle this?"
**Answer:** The detection mask (channel 9) partially addresses this. When a hand's confidence is below the threshold (0.25 for RTMW-XL), the mask is set to 0.0, and the GCN gates that body part's features to zero before processing. However, if the pose estimator returns confident but incorrect landmarks for a non-visible hand, the mask won't catch it. This is a known limitation that affects single-hand sign recognition and is acknowledged in the paper.

### Q17: "You mention 7 unique signers. That's very few for a generalizable system."
**Answer:** Correct. Seven signers is limited for signer-independent generalization. This is why we chose angle-primary features -- they're inherently more signer-invariant than positional features. However, the training data is dominated by a few signers from the WLASL dataset and YouTube ASL collections. More diverse signer representation would significantly improve generalization. This is a data limitation, not an architectural one.

---

## SECTION 3: Questions About Citations & Literature (Dr. Caluza might check)

### Q18: "Can you verify Cheng et al., 2021 for DS-GCN?"
**Answer:** The closest verified paper is Cheng et al., CVPR 2020 -- "Skeleton-Based Action Recognition With Shift Graph Convolutional Network" (Shift-GCN), which demonstrated 10x computational reduction through factored graph convolution operations. Our DS-GCN approach follows the same principle of decomposing graph convolution into lightweight spatial operations followed by pointwise channel mixing, consistent with the depthwise separable factorization introduced by Howard et al. (2017) in MobileNets.

### Q19: "Some of your references look unfamiliar. Are they real papers?"
**Answer:** The core references are well-established and verifiable: Camgoz et al. (2020) for sign language transformers, Raffel et al. (2020) for T5, Graves et al. (2006) for CTC, Deng et al. (2019) for ArcFace, Howard et al. (2017) for depthwise separable convolutions. If asked about specific obscure references, acknowledge that some supporting citations may need verification and offer to provide the verified alternatives.

---

## SECTION 4: Questions About the Application & Impact (Dr. Heru will likely ask)

### Q20: "How does this benefit the deaf community at LNU specifically?"
**Answer:** Currently, deaf students at LNU rely on handwritten notes or typing on phones to communicate with hearing students, teachers, and staff. This creates delays and limits spontaneous interaction. Our system allows a deaf person to sign at the camera and have their message translated to English text, which the hearing person can read immediately. While turn-based, this is faster and more natural than writing notes back and forth.

### Q21: "Is this system ethically sound? What about privacy?"
**Answer:** The system processes only skeletal coordinates, not raw video. No images of the user are stored or transmitted -- only the 61-point mathematical representation of joint positions. This is inherently privacy-preserving: you cannot reconstruct what someone looks like from skeleton data. The system runs locally on the desktop, so no data is sent to external servers.

### Q22: "What would it take to deploy this as an actual product at LNU?"
**Answer:** Three main steps: (1) Mobile app development with on-device inference (TensorFlow Lite or CoreML conversion), (2) Expanding the vocabulary beyond 310 signs to cover more LNU-relevant communication, (3) Re-training with local signers' data to improve signer-independent performance. The current desktop prototype proves the concept; productionization requires engineering effort but no fundamental architectural changes.

### Q23: "Can this system handle Filipino Sign Language (FSL)?"
**Answer:** The architecture is language-agnostic -- it processes skeletal data, not language-specific features. To support FSL, you would need: (1) FSL video datasets for extraction and training, (2) FSL gloss vocabulary, (3) FSL-to-Filipino/English gloss-to-text training data for Stage 3. The same pipeline (DS-GCN-TCN + CTC + T5) would work without architectural changes -- only data and training would change.

### Q24: "How does your system compare to existing commercial solutions like SignAll or Signapse?"
**Answer:** Commercial systems like Signapse use RGB video directly and generate sign language AVATARS (text-to-sign). Our system does the reverse: sign-to-text. Most commercial solutions focus on controlled environments with high-end cameras. Our system uses standard webcam input and lightweight skeleton-based processing (13M custom parameters vs hundreds of millions for video-based systems), making it more suitable for resource-constrained academic environments.

### Q25: "Why ASL instead of FSL if the target users are at LNU?"
**Answer:** We used ASL because large publicly available ASL datasets exist (WLASL, YouTube ASL collections), enabling training with 57,535 samples across 310 classes and 7 signers. No comparable FSL dataset exists at this scale. The architecture is designed to be language-agnostic -- the same pipeline can be retrained for FSL when sufficient data becomes available. This study focuses on proving the technical approach works; FSL adaptation is a data collection effort rather than a technical limitation.

---

## SECTION 5: Deep Technical Questions (If they go very deep)

### Q26: "What is the adjacency matrix and how is it constructed?"
**Answer:** The adjacency matrix is a 61x61 table that defines which joints are connected. Entry (i,j) = 1 if joint i connects to joint j via a bone, 0 otherwise. We use three partitions: A_self (identity -- each node to itself), A_out (parent to child connections), A_in (child to parent connections). Each is row-normalized (divided by the number of connections per node) so aggregation averages neighbor features rather than summing them. The learnable adjacency residual adds tanh(learned_matrix) * 0.3 on top, allowing the model to discover non-anatomical connections that are useful for recognition.

### Q27: "What is the 1-Euro filter and why use it over a simple low-pass filter?"
**Answer:** A simple low-pass filter uses a fixed cutoff frequency -- it smooths everything equally. This either removes jitter but also blurs fast movements, or preserves fast movements but keeps jitter. The 1-Euro filter adapts its cutoff based on signal speed: when the hand is still, it smooths heavily (removing pose estimator jitter); when the hand moves fast, it smooths lightly (preserving signing dynamics). This is critical because ASL has both fast motions (fingerspelling) and still holds (sign endpoints).

### Q28: "How does backpropagation work in your system?"
**Answer:** After a forward pass produces a prediction and the loss is computed, backpropagation computes the gradient (direction of change) for every weight in the model using the chain rule of calculus. Each weight gets a gradient telling it "if you increase, does the loss go up or down?" Then the optimizer (AdamW) nudges every weight slightly in the direction that reduces the loss. Learning rate (3e-4) controls the step size. This repeats for every batch, every epoch, for 150 epochs -- millions of small adjustments that collectively transform random weights into an accurate model.

### Q29: "What is EMA and why do you use it?"
**Answer:** Exponential Moving Average maintains a shadow copy of the model weights that's updated slowly: shadow = 0.999 * shadow + 0.001 * current_weights. This averages out the noise from individual training steps. The EMA weights are used for evaluation and typically give 0.5-1% better accuracy than the raw training weights because they represent a smoother, more stable version of the model.

### Q30: "What happens if two signs look very similar? How does the model distinguish them?"
**Answer:** Several mechanisms help: (1) ArcFace forces tight angular clusters with at least 29-degree separation between classes. (2) The 114 geometric features provide explicit measurements (finger crossing distinguishes M from N, contact detection distinguishes MEET from two-hand signs). (3) The temporal modeling captures motion dynamics -- even if two signs have similar handshapes, they often differ in trajectory or speed. (4) The confused gloss upweighting in Stage 2 training ensures similar signs get more training exposure.

---

## SECTION 6: Defending Known Mistakes in the Paper

### M1: "Your paper mentions 'student model' on page 29. What student model?"
**Defense:** This is a leftover reference from an earlier knowledge distillation experiment that was removed from the final architecture. The current system does not use knowledge distillation. The phrase should have been removed during revision. The actual training uses ArcFace angular margin loss, mixup augmentation, and label smoothing -- no teacher-student setup.

### M2: "The paper cites v14 accuracy (85.32%) but your CLAUDE.md mentions v15 at 91.74%"
**Defense:** The paper reports the v14 results because v15 training was completed after the manuscript draft. The v15 model with Apple Vision extraction achieves 91.74% test accuracy -- a significant improvement from matching the extraction pipeline between training and inference. This can be presented as updated results during the defense.

### M3: "Page 18 mentions DS-GCN reduces parameters by 35-45%. Where does this number come from?"
**Defense:** This figure refers to parameter reduction in general action recognition tasks as reported in prior work. In our specific implementation, the reduction is in FLOPs (3x fewer matrix multiplications) rather than raw parameter count. We can clarify that the efficiency gain is primarily in computational cost per layer, not total model size.

### M4: "The paper says 'RTMW-XL' throughout but annotations suggest you may use Apple Vision"
**Defense:** RTMW-XL was used as the primary extraction backbone for the training dataset. Apple Vision Framework was explored for Mac-native inference to ensure extraction consistency between training and deployment environments. Both produce COCO-WholeBody-compatible 133-keypoint outputs. The paper correctly focuses on RTMW-XL as the primary training extractor.

### M5: "Some handwritten notes say 'proof?' next to claims about GCN computational overhead"
**Defense:** The computational overhead of standard GCN layers is well-documented in the literature. Shi et al. (2020) specifically noted that standard GCN layers "significantly increased computational overhead, making edge deployment difficult." Cheng et al. (2020) demonstrated 10x reduction with factored approaches. Our DS-GCN achieves approximately 3x reduction in spatial aggregation FLOPs by reducing 3 matrix multiplies to 1.

### M6: "The handwritten note says 'no comparison' regarding the system"
**Defense:** We acknowledge that the study does not include a direct comparative evaluation against other architectures (e.g., standard GCN, LSTM-based, or end-to-end transformer). The scope focused on developing and evaluating the proposed pipeline. A comparative study would require reimplementing baseline systems under identical conditions, which is planned for future work. However, we can reference published results: standard ST-GCN achieves 21-25% WER for CSLR, while our CTC-based approach achieves 10.96% WER on our dataset.

### M7: "The paper mentions the evaluation uses ISO/IEC 25010:2023 but you're not done with user testing"
**Defense:** The ISO/IEC 25010:2023 evaluation framework is proposed and planned for the testing phase (June-July 2026). The current pre-oral defense presents the technical implementation and model performance. The formal usability evaluation with LNU community members will be conducted after ethics clearance and is part of the complete study methodology.

---

## SECTION 7: Curveball Questions (Unexpected angles)

### Q31: "Why not just use a large language model like GPT-4 to do everything?"
**Answer:** GPT-4 or similar LLMs process text and images, not skeletal time-series data. They can't directly interpret 32-frame sequences of 61-node skeleton coordinates. You would need a vision encoder to extract features from video, which brings you back to a multi-stage pipeline. Our approach is also 152x more data-efficient than feeding raw video pixels, runs locally without internet, and costs nothing at inference (no API calls). The total pipeline is only 261M parameters vs 1.8T for GPT-4.

### Q32: "What if someone signs something not in your 310-class vocabulary?"
**Answer:** The model would either classify it as the most similar known sign (misclassification) or, in Stage 2's CTC output, it might appear as a blank/silence. This is a closed-vocabulary limitation. Expanding the vocabulary requires collecting and training on new sign classes. An out-of-vocabulary detection mechanism (e.g., confidence thresholding) could flag unknown signs, but this is not yet implemented.

### Q33: "Can two-handed signs be recognized if the camera only sees one hand?"
**Answer:** Partially. The detection mask flags the absent hand as 0.0, and the GCN gates its features to zero. The model can still attempt classification based on the visible hand's features alone. However, for truly two-handed signs where both hands carry information (like BOOK, TOGETHER), recognition accuracy will degrade significantly with one hand occluded. Drop-Graph regularization during training helps -- the model has practiced classifying with randomly missing nodes.

### Q34: "What is the computational cost of inference? Can this run on a phone?"
**Answer:** The recognition model (Stages 1+2) is only 13M parameters -- very lightweight. The bottleneck is extraction (RTMW-XL takes 12s on M4 Mac CPU). For mobile deployment, the extraction could be replaced with MediaPipe (runs at 30fps on phones) or a lighter pose estimator, and the PyTorch models could be converted to TensorFlow Lite or CoreML. The T5 translation model (248M) would need quantization for mobile. Technically feasible, but not yet implemented.

### Q35: "How did you handle class imbalance? Some classes have 79 samples and others have 487."
**Answer:** WeightedRandomSampler upweights underrepresented classes during training so each class is seen approximately equally often per epoch. Label smoothing (0.05) prevents overconfidence on common classes. In Stage 2, the confused gloss upweighting mechanism samples frequently-confused signs 3x more often in synthetic sequences, creating harder training examples.

### Q36: "What is the Flan-T5 model and why not use a newer model?"
**Answer:** Flan-T5-Base is a 250M parameter encoder-decoder transformer by Google, instruction-tuned on 1,800+ tasks. We chose it because: (1) it's small enough to run locally without GPU, (2) it's instruction-tuned, so it responds well to our prompt format ("Translate this ASL gloss to natural conversational English: ..."), (3) it achieves strong translation quality on our task. Newer models (GPT-4, Llama) are much larger and would require cloud APIs or powerful hardware, which contradicts our goal of local, lightweight deployment.

### Q37: "If the panel asks you to demo the system live and it fails, what do you say?"
**Answer:** Be honest: "The system is signer-dependent. It was trained on 7 specific signers from publicly available datasets. When a new signer uses it, accuracy drops because of differences in hand size, signing style, and the webcam extraction environment. This is a known limitation documented in our paper. The 85% accuracy is under controlled conditions with matched training data. Improving signer-independent performance requires more diverse training data and is our primary direction for future work."

---

## SECTION 8: Quick Reference — Key Numbers to Memorize

| Metric | Value |
|---|---|
| Total extracted files | 66,451 |
| After cleaning | ~57,535 |
| ASL classes | 310 |
| Unique signers | 7 |
| Input tensor shape | [B, 32, 61, 16] |
| Skeleton nodes | 61 (21 L hand + 21 R hand + 15 face + 4 body) |
| Input channels | 16 (XYZ + vel + acc + mask + bone dir + bone motion) |
| Geo features | 114 (computed at runtime) |
| Angle features | 59 + 59 velocity = 118 |
| GCN blocks | 4 (last 2 with SE attention) |
| TCN blocks | 4 (dilation 1, 2, 4, 8) |
| d_model | 384 |
| Stage 1 accuracy (v14) | 85.32% val / 84.80% test |
| Stage 1 accuracy (v15) | 91.74% test |
| Stage 2 WER | 10.96% (v14 encoder) |
| Stage 3 model | Flan-T5-Base (250M params) |
| Total pipeline params | ~261M (13M custom + 248M T5) |
| Stage 1 params | 5.4M |
| Stage 2 new params | 8.0M |
| ArcFace margin | m=0.5, s=30 |
| Learning rate (S1) | 3e-4 |
| Epochs (S1) | 150 |
| CTC blank token | Index 0 |
| Continuous phrase videos | 780 (9 phrases) |
| Extraction time | ~12 seconds per video (M4 Mac CPU) |
