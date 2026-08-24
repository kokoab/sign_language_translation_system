# SLT Pipeline: Complete Technical Dissection

## Overview

This pipeline translates raw webcam video of a person signing ASL into natural English text through four stages:

```
Video -> [Stage 0: Extraction] -> [32, 61, 16] tensor
      -> [Stage 1: Classification] -> sign class (310 classes)
      -> [Stage 2: CTC Recognition] -> gloss sequence ("HELLO HOW YOU")
      -> [Stage 3: Translation] -> English ("Hello, how are you?")
```

**Dataset:** 310 ASL sign classes, 57,535 total samples, 61-node skeleton (42 hand + 15 face + 4 body), 16-channel features (XYZ + velocity + acceleration + mask + bone direction + bone motion) stored directly in the .npy files.

---

## Stage 0 -- Extraction (Video -> Pose Tensor)

**Goal:** Convert raw video frames into a compact, normalized skeleton tensor that captures hand shape, hand motion, and face/body reference points.

**Core files:** `src/extract.py`, `extract_batch_fast_v2.py`, `extract_batch_rtmlib.py`

### The Pose Estimator: RTMW-XL

RTMW-XL (Real-Time Models for Wholebody) is a top-down pose estimation model from the MMPose ecosystem. It was trained on 14 datasets ("Cocktail14") and distilled from a larger teacher model. It outputs **133 COCO-WholeBody keypoints** per person per frame:

| Keypoint Range | Body Part | Count |
|---|---|---|
| 0-16 | Body (torso, head, limbs) | 17 |
| 17-22 | Feet | 6 |
| 23-90 | Face (68 landmarks) | 68 |
| 91-111 | Left Hand | 21 |
| 112-132 | Right Hand | 21 |

From these 133 keypoints, the pipeline selects **61 points**: 21 left hand + 21 right hand + 15 face + 4 body = 61 nodes.

- **Hands (0-41):** Each hand follows the standard 21-point hand skeleton topology: wrist (node 0), then 4 finger chains of 4 joints each (MCP, PIP/IP, DIP, TIP for thumb through pinky). Inter-finger connections link adjacent MCP joints (index-middle, middle-ring, ring-pinky). Left hand = nodes 0-20, Right hand = nodes 21-41.
- **Face (42-56):** 15 face points -- nose (42), chin (43), forehead (44), left/right ears (45-46), left/right mouth corners (47-48), upper/lower lip (49-50), inner/outer left eyebrow (51-52), inner/outer right eyebrow (53-54), left/right eyes (55-56).
- **Body (57-60):** 4 body points -- left shoulder (57), right shoulder (58), left elbow (59), right elbow (60). These connect to wrists via edges (elbow->wrist), giving the model arm chain context.

**Multi-person handling:** When multiple people appear in frame, the pipeline picks the **largest person** by body bounding box area (the most prominent signer).

### Raw Coordinates to Final Tensor -- The Full Pipeline

Each video goes through this exact sequence:

#### Step 1: Frame Extraction & Preprocessing

Frames are decoded from the video file. Before pose estimation, each frame undergoes image preprocessing:
- **Adaptive gamma correction** -- adjusts overall brightness
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) -- normalizes local contrast so pose estimation works in poor lighting
- **Bilateral denoising** -- edge-preserving smoothing that reduces camera noise while keeping landmark edges sharp
- **Unsharp masking** -- edge enhancement to make joint boundaries more detectable

This ensures the pose estimator gets clean, well-exposed input regardless of webcam quality or lighting conditions.

#### Step 2: RTMW-XL Inference

Frames are processed in batches of 32 on GPU. For each frame, the model outputs 133 keypoints with XY pixel coordinates and confidence scores. The pipeline normalizes pixel coordinates to `[0, 1]` range by dividing by image width and height: `x_norm = x_pixel / width`, `y_norm = y_pixel / height`.

Per-keypoint confidence scores determine which detections are valid. The hand confidence threshold is 0.25 (RTMW-XL's high quality allows a lower threshold than other models). Face landmarks require 0.3 confidence.

#### Step 2b: Z Depth Estimation

RTMW-XL outputs 2D keypoints (X, Y only). The pipeline estimates a relative Z (depth) coordinate using **perspective projection from anatomical scale invariants**:

The key insight is that anatomically rigid structures (bones) have fixed real-world lengths. When a hand moves closer to the camera, its apparent 2D size increases; when it moves away, the apparent size shrinks. The pipeline exploits this:

1. Compute the **reference palm length** -- the median wrist-to-middle-MCP distance across the entire sequence (this is the "true" palm size, averaged over all frames to cancel out noise)
2. At each frame, measure the **observed palm length** in 2D pixel-normalized coordinates
3. Estimate relative depth: `Z = reference_palm_length / observed_palm_length`

When the observed palm appears larger than the reference, the hand is closer (Z < 0). When it appears smaller, the hand is farther away (Z > 0). This gives per-frame, per-hand relative depth.

For face nodes, the same principle applies using the inter-ear distance as the scale invariant. For body nodes (shoulders, elbows), depth is interpolated from the face and hand estimates.

This is not absolute metric depth -- it's a relative depth signal normalized alongside X and Y during the normalization step. The model uses it as a soft cue for hand orientation and spatial relationships (e.g., one hand in front of the other for signs like BEHIND, FRONT).

#### Step 3: Temporal Coherence Rejection

Before interpolation, the pipeline removes **false detection jumps** -- frames where the wrist teleports an unrealistic distance between consecutive detections. Using the wrist as a proxy for the whole hand, any frame with a displacement > 15% of the coordinate space is rejected (`jump_threshold=0.15`). This prevents single-frame misdetections from corrupting the interpolation.

**"Temporal coherence"** means landmarks should move smoothly through time. A hand can't teleport from one side of the frame to the other in 1/30th of a second.

#### Step 4: 1-Euro Adaptive Low-Pass Filter

The **one-euro filter** (Casiez et al. 2012) is an adaptive low-pass filter with a key property: it applies **heavy smoothing when the hand is still** (removing pose estimator jitter) but **light smoothing when the hand moves fast** (preserving the dynamics of signing).

How it works mathematically:
- It maintains a derivative estimate `dx[i]` filtered with a fixed cutoff `d_cutoff`
- The actual cutoff frequency adapts: `cutoff = min_cutoff + beta * |dx[i]|`
- When speed `|dx|` is low -> cutoff is low -> heavy smoothing (jitter removal)
- When speed `|dx|` is high -> cutoff is high -> minimal smoothing (preserve fast motion)
- The smoothing itself: `y[i] = alpha * x[i] + (1 - alpha) * y[i-1]` where `alpha = 2*pi*cutoff*dt / (2*pi*cutoff*dt + 1)`

Parameters: `min_cutoff=1.0` (baseline smoothing), `beta=0.007` (speed sensitivity), `d_cutoff=1.0` (derivative smoothing), `t_e=1/30` (frame interval at 30fps).

#### Step 5: Interpolation & Gap Filling

Not every frame produces a valid detection for every hand. The pipeline tracks which frame indices have valid detections (`l_valid`, `r_valid`, `face_valid`). Missing frames are filled using **linear interpolation** between valid detections. If only one valid frame exists, it's replicated across all frames.

The interpolation works on flattened coordinates: a hand's 21 x 3 = 63 values are interpolated independently across the time axis using `numpy.interp`.

#### Step 6: Bone Length Stabilization

Even after filtering, pose estimators produce slightly different bone lengths frame-to-frame (e.g., the thumb might appear 5% longer in one frame vs. another). This is physically impossible -- bone lengths don't change.

The pipeline computes the **median bone length** across the entire sequence for each bone pair (e.g., wrist->thumb_mcp, thumb_mcp->thumb_ip, etc.), then rescales each frame so every bone has its median length while preserving direction:

```python
unit = vecs / (lengths + 1e-8)
xyz[:, child] = xyz[:, parent] + unit * median_len
```

This uses 20 bone pairs per hand following the standard hand skeleton connectivity (5 fingers x 4 bones each).

#### Step 7: Normalization (`normalize_sequence`)

The 3D coordinates are centered and scaled:

1. **Centering:** Compute the median wrist position across all valid frames of all active hands. Subtract this center from all hand coordinates AND face landmarks. This makes the representation **translation-invariant** -- the sign "HELLO" produces the same tensor regardless of where the signer is in the frame.

2. **Scaling:** Compute the median wrist-to-middle-MCP bone length (the palm length) across all active hands. Divide all coordinates by this value. This makes the representation **scale-invariant** -- the sign looks the same whether the person is close to or far from the camera.

#### Step 8: Temporal Resampling

Every video clip, regardless of original frame count or video framerate, is resampled to exactly **32 frames** using linear interpolation. This creates a fixed-length temporal representation. If a clip has 45 frames, it's compressed; if it has 20, it's stretched.

The resampling uses vectorized linear interpolation: source timestamps are mapped to `np.linspace(0, 1, T_original)` and target timestamps to `np.linspace(0, 1, 32)`.

#### Step 9: Kinematics Computation (`compute_kinematics_batch`)

From the 32 x 61 x 3 (XYZ) array, the pipeline computes:

- **Velocity** (3 channels): Rate of change of position. Computed via **Savitzky-Golay filter** (window=7, polynomial order=2, first derivative) when scipy is available, otherwise via central difference: `vel[t] = (xyz[t+1] - xyz[t-1]) / 2`. Edge frames copy their neighbors. Savitzky-Golay fits a local polynomial to a sliding window and differentiates analytically -- much smoother than finite differences on noisy data.

- **Acceleration** (3 channels): Rate of change of velocity. Same method, second derivative.

- **Mask** (1 channel): Binary per-node-per-frame flag. 1.0 = this node was detected (or interpolated from detections), 0.0 = never detected for this body part. The mask is set per body part: all 21 left hand nodes share a flag, all 21 right hand nodes share a flag, all face/body nodes share their respective flags.

#### Step 10: Final Output

The saved tensor is **`[32, 61, 16]` float16** where:

| Channels | Meaning | Count |
|---|---|---|
| 0-2 | XYZ position (normalized, centered; Z is estimated relative depth from perspective projection) | 3 |
| 3-5 | Velocity (dx, dy, dz per frame) | 3 |
| 6-8 | Acceleration (dvx, dvy, dvz per frame) | 3 |
| 9 | Detection mask (1.0 = present, 0.0 = absent) | 1 |
| 10-12 | Bone direction (parent->child displacement vector) | 3 |
| 13-15 | Bone motion (temporal derivative of bone direction) | 3 |

Node indices (61 total):

| Indices | Body Part | Count |
|---|---|---|
| 0-20 | Left Hand (wrist + 4 fingers x 5 joints) | 21 |
| 21-41 | Right Hand (same layout, offset by 21) | 21 |
| 42-56 | Face (15 points: nose, chin, forehead, ears, mouth, eyebrows, eyes) | 15 |
| 57-60 | Body (L/R shoulders, L/R elbows) | 4 |

Bone features (channels 10-15) are computed during extraction and stored directly in the .npy files, not at load time. See the Bone Features section under Stage 1 for details on what these channels represent.

Each `.npy` file represents one isolated sign and is stored in a folder named by its label (e.g., `ASL_landmarks_float16/HELLO/HELLO_0062d3ea.npy`).

---

## Stage 1 -- Isolated Sign Classification

**Goal:** Given a single 32-frame clip of one sign, classify it into one of 310 ASL sign classes.

**Core files:** `src/train_stage_1.py`, `src/model_v14.py`, `src/train_v15.py`

### Input: [B, 32, 61, 16]

The `[32, 61, 16]` tensor is loaded directly with all 16 channels already present (bone features are computed during extraction and stored in the .npy files):

#### Bone Features (6 channels, indices 10-15)

- **Bone direction** (channels 10-12): For each bone pair (parent->child), the vector `xyz[child] - xyz[parent]`. This captures **hand shape independently of position** -- a bent finger has a different bone direction vector than an extended finger, regardless of where the hand is in space. For face nodes, bone vectors are relative to the nose. For body nodes, elbow vectors are relative to shoulders. 20 bone pairs per hand x 2 hands + face + body bones.

- **Bone motion** (channels 13-15): Central difference of bone direction vectors across time: `bone_motion[t] = (bone[t+1] - bone[t-1]) / 2`. This captures **how the hand shape changes over time** -- a closing fist has different bone motion than an opening hand.

Each node has 16 features: `[xyz(3), velocity(3), acceleration(3), mask(1), bone_direction(3), bone_motion(3)]`. All 16 channels are stored in the .npy files and loaded directly -- no additional feature computation is needed at training time.

### Architecture: SLTStage1V14 (DS-GCN-TCN with Angle-Primary Features)

The model has four components: **DSGCNEncoderV14** (spatial) -> **Angle Features** (signer-invariant) -> **TemporalTCN** (temporal) -> **ArcFace Head** (classification).

#### DSGCNEncoderV14 -- Depthwise Separable Graph Convolution Network

##### What is a GCN (Graph Convolutional Network)?

A regular CNN assumes data lives on a grid (like pixels in an image). But a hand skeleton is a **graph**: nodes (joints) connected by edges (bones). A GCN generalizes convolution to work on arbitrary graph structures. Instead of convolving over a local pixel neighborhood, it convolves over a node's **graph neighbors** -- the joints connected to it by bones.

For each node, a GCN aggregates the features of its neighbors (as defined by the adjacency matrix), applies a linear transformation, and produces an updated feature vector. Multiple GCN layers allow information to propagate across longer paths in the graph (e.g., from fingertip to wrist requires traversing 4 edges, so 4 GCN layers would be needed).

##### What makes it "Depthwise Separable" (DS)?

A standard multi-partition GCN (e.g., ST-GCN) uses a separate full weight matrix for each adjacency partition -- 3 partitions means 3 expensive matrix multiplications per layer. DS-GCN factorizes this into two cheaper steps:

1. **Depthwise (spatial aggregation):** Each adjacency partition is weighted by a lightweight per-channel scaling vector, not a full matrix. The graph has 3 adjacency matrices:
   - **A_self:** Identity -- each node keeps its own features
   - **A_out:** Information flows from parent to child (e.g., wrist -> fingertip direction)
   - **A_in:** Information flows from child to parent (e.g., fingertip state aggregated at wrist)

   Each partition aggregates neighbor features separately using `torch.einsum('knm,btnc->kbtnc', A, x)`, then scales by learnable weights `dw_weights[k]` (a vector of size `[C_in]`, not a matrix). This keeps the three directional signals separated while using minimal parameters.

2. **Pointwise (channel mixing):** The 3 partitions are concatenated (tripling the channel count) and projected through a single shared linear layer: `Linear(3*C_in, C_out)`. This is the only full matrix multiply -- it happens once, not three times.

**Why this is more efficient:** A standard GCN does 3 full matrix multiplies (one per partition). DS-GCN does 3 cheap vector scalings + 1 matrix multiply. The directional awareness is preserved (each partition stays separate until the pointwise step), but the expensive computation is reduced from 3x to 1x.

Each adjacency matrix is row-normalized (divided by node degree) so aggregation averages rather than sums neighbor features.

##### Learnable Adjacency Residual

On top of the fixed anatomical graph, each `DSGCNBlock` has a learnable residual:

```python
A_eff = A + tanh(adj_residual) * 0.3
```

This allows the network to discover that, for instance, the thumb tip should also attend to the pinky tip (not an anatomical connection, but useful for recognizing a "pinch" handshape). The `tanh` bounds the residual to [-0.3, 0.3], preventing the learned graph from completely overriding the anatomical prior.

##### Temporal Convolution within DSGCNBlock

After spatial graph convolution, each block also applies a **depthwise 1D convolution along the time axis** (kernel sizes 3, 5, or 7) with GroupNorm. This captures local temporal patterns within each node -- e.g., a fingertip oscillating or a wrist following a circular path.

##### Drop-Graph Regularization

During training, random nodes are masked with probability `node_drop_rate=0.05`, forcing the network to not over-rely on any single joint. This is like dropout but at the graph node level. A safety check ensures at least one node per body part group (left hand, right hand, face, body) survives.

##### Squeeze-and-Excitation (SE) Attention

The last two GCN blocks include **SE attention** (Hu et al., CVPR 2018) -- a channel-wise recalibration mechanism. It computes a per-channel importance weight by:
1. Global average pooling across nodes and frames -> one value per channel
2. Two FC layers (bottleneck with reduction=4) -> sigmoid -> importance weight per channel
3. Multiply each channel by its importance weight

This lets the network learn "for this input, channels encoding thumb position matter more than channels encoding pinky position" and scale accordingly.

##### The Full Encoder Stack (4 GCN Blocks + TCN)

```
input_norm(LayerNorm) -> input_proj(Linear 16->96, LayerNorm, GELU)
-> GCN Block 1 (96->192, kernel=3)
-> GCN Block 2 (192->384, kernel=5)
-> GCN Block 3 with SE (384->384, kernel=5)
-> GCN Block 4 with SE (384->384, kernel=7)
-> Node Attention (softmax over 61 nodes -> weighted sum -> [B, 32, 384])
-> Concatenate with Angle Features [B, 32, 118]
-> angle_proj(Linear 384+118 -> 384)
-> TemporalTCN (4 dilated blocks)
-> output [B, 32, 384]
```

**Node Attention:** After 4 GCN blocks, each frame has features for all 61 nodes. Node attention learns which joints matter most for each frame: `attn = softmax(Linear(d -> d/4) -> GELU -> Linear(d/4 -> 1))` produces a weight per node, then the weighted sum collapses the node dimension: `[B, T, 61, 384] -> [B, T, 384]`.

#### Angle-Primary Features (59 features + 59 velocities = 118 total)

The key architectural insight of v14/v15: instead of relying on XYZ coordinates as the primary discriminative signal, the model computes **angle-based features** that are inherently **signer-invariant**.

**Why?** Cosine similarity tests showed:
- XYZ features: 0.63 similarity between webcam and training data (signer-dependent)
- Angle features: 0.90 similarity (signer-invariant)
- Curl ratios: 0.93 similarity (nearly identical across signers)

The 59 angle features are computed at runtime from the XYZ coordinates:

| Feature Group | Count | What It Captures |
|---|---|---|
| **Joint angles** (per hand) | 15x2=30 | Angle at each finger joint (3 per finger x 5 fingers). Computed via `arccos(v1 dot v2 / |v1||v2|)`. Completely signer-invariant -- the same handshape produces the same angles regardless of hand size or arm length. |
| **Curl ratios** (per hand) | 5x2=10 | `MCP->TIP distance / MCP->PIP distance`. Ratio > 1 = extended finger, < 1 = curled. Insensitive to hand size. |
| **Finger spread** (per hand) | 3x2=6 | Angle between adjacent MCP directions from wrist. Distinguishes spread hand (like "5") from closed hand (fist). |
| **Palm normals** (per hand) | 3x2=6 | Cross product of (wrist->index_MCP) x (wrist->pinky_MCP), normalized. 3 components (x,y,z) = palm facing direction. |
| **Wrist orientation** (per hand) | 2x2=4 | Palm normal dotted with canonical up and forward directions. Palm up/down or forward/backward. |
| **Inter-hand distance + direction** | 2 | Normalized wrist-to-wrist distance + vertical relative direction. |
| **Hand symmetry** | 1 | Cosine similarity between left and right hand angle vectors. High for symmetric signs (BOOK), low for asymmetric (HELLO). |

**Total: 59 features.**

Additionally, the **temporal derivative** (velocity) of all 59 features is computed, giving 59 more features for a total of **118 angle features per frame**. These capture how the angles change over time -- e.g., a closing fist has decreasing joint angles.

The angle features are concatenated with the GCN output and projected: `Linear(384 + 118, 384)`. The GCN still operates on per-node features (XYZ + bone + mask) for spatial relationships, but the angle features become the **primary discriminative signal**.

#### TemporalTCN (4 Dilated Blocks)

After the GCN handles spatial relationships and angle features provide signer-invariant hand shape, the **TemporalTCN** models temporal patterns across the 32 frames using dilated 1D convolutions:

```
Block 1: dilation=1  (sees 3 neighboring frames)
Block 2: dilation=2  (sees 5 frames)
Block 3: dilation=4  (sees 9 frames)
Block 4: dilation=8  (sees 17 frames)
```

The exponentially increasing dilation means each block sees a wider temporal window without adding parameters. By block 4, each frame has context from 17 frames -- over half the 32-frame clip. Each block uses: `Conv1d -> GroupNorm -> GELU -> Conv1d -> GroupNorm -> GELU -> residual connection`.

**DropPath** (stochastic depth) is applied with linearly increasing rates across the TCN blocks, randomly skipping entire blocks during training for regularization.

**Why TCN instead of Transformer?** With the strong angle-primary features, the temporal model doesn't need global self-attention. TCN is more efficient (linear compute vs quadratic for Transformer attention) and the dilated receptive field of 17 frames covers enough temporal context for a 32-frame sign clip.

#### ArcFace Head

**ArcFace** (Deng et al. CVPR 2019) adds an **angular margin penalty** to the classification loss, borrowed from face recognition. Instead of standard softmax:

1. L2-normalizes both the feature vector and the weight matrix
2. Computes cosine similarity: `cos(theta) = normalize(features) dot normalize(weights)`
3. For the correct class, adds a margin: `cos(theta + m)` instead of `cos(theta)`
4. Scales by a temperature: `s * cos(theta + m)`

This pushes the model to learn features where same-class samples are tightly clustered and different classes are widely separated in angular space. Think of it as forcing a minimum "angular gap" between decision boundaries -- features must be more confidently correct to get high scores.

The margin `m=0.5` (~29 degrees) and scale `s=30` are standard values from the face recognition literature.

**Frame attention** pools the 32 frame embeddings into a single vector before classification: `softmax(Linear(384 -> 96) -> GELU -> Linear(96 -> 1))` produces per-frame weights, then the weighted sum gives `[B, 384]`. This is critical because the "stroke" (main motion) matters more than preparation/retraction phases.

**Warmup schedule:** ArcFace can collapse early in training (the model hasn't learned meaningful features yet, so adding angular margin causes chaos). The warmup prevents this:
- Epochs 1-10: pure cross-entropy (m=0, s=1) -- learn basic features first
- Epochs 11-30: linearly ramp m from 0->0.5 and s from 10->30 -- gradually introduce margin
- Epochs 31+: full ArcFace

### Loss Function

**Cross-entropy with label smoothing** (smoothing=0.05): Instead of hard targets `[0, 0, 1, 0, ...]`, use soft targets `[0.05/310, 0.05/310, 0.95 + 0.05/310, ...]`. This prevents the model from becoming overconfident and improves generalization.

Focal loss (gamma=0.0) is **disabled** -- it was found to hurt accuracy on this dataset.

### Training Details

- **Optimizer:** AdamW, lr=3e-4, weight decay
- **Schedule:** Cosine decay (NO warm restarts -- these caused training instability)
- **Gradient accumulation:** 4 steps, effective batch = 256 x 4 = 1024
- **EMA (Exponential Moving Average):** A moving average of model weights is maintained (`decay=0.999`) and used for evaluation. EMA smooths out training noise and typically gives 0.5-1% better accuracy.
- **WeightedRandomSampler:** Upweights underrepresented classes to handle the 6.2x class imbalance (TEACHER has 79 samples vs. I has 487).
- **Epochs:** 150, patience=25

### Output

310-class logits -> softmax -> predicted sign label. **Stage 1 achieves 91.74% Top-1 accuracy** on the test set (v15 model with Apple Vision extraction).

---

## Stage 2 -- Continuous Sign Recognition (CTC)

**Goal:** Given a variable-length sequence of multiple signs (not segmented), produce the ordered sequence of gloss labels.

**Core file:** `src/train_stage_2.py`

This is the hardest stage. In Stage 1, each clip is one sign. In real conversation, signs flow together without clear boundaries -- this is **coarticulation**, where the end of one sign blends into the beginning of the next (the same phenomenon as connected speech in spoken language). The model must figure out where signs begin and end, and what they are, simultaneously.

### Key Term: Coarticulation

When you sign "HELLO HOW YOU" in continuous ASL, the hand doesn't return to a neutral position between signs. The ending handshape of HELLO starts blending into the beginning handshape of HOW while the hand is still moving. This creates transition frames that don't match any isolated sign -- they're in-between states. Stage 2 must handle this.

### Input: [B, T, 61, 16]

Variable-length sequences where T is a multiple of 32. Each 32-frame chunk represents roughly one sign's duration. The 16 channels are the same as Stage 1 (all stored in the .npy files).

### Architecture: SLTStage2CTC

```
Input [B, T, 61, 16]
  |
  v
Split into 32-frame clips -> [total_clips, 32, 61, 16]
  |
  v
Frozen Stage 1 DSGCNEncoderV14 (4 GCN blocks + angle features + TCN)
  -> [total_clips, 32, 384] per-frame embeddings
  |
  v
MultiScaleTCN (3 parallel convolution branches + pool)
  -> [total_clips, 4, 384] (32 frames compressed to 4 tokens per clip)
  |
  v
Reshape to per-sample sequences -> [B, num_clips*4, 384]
  |
  v
SequenceTransformer (4 layers)
  -> [B, num_clips*4, 384] cross-clip temporal modeling
  |
  v
Linear(384, 311) -> CTC logits (310 classes + 1 blank)
  |
  v
CTC decoding -> gloss sequence
```

#### Frozen Stage 1 Encoder (first 30 epochs)

The DSGCNEncoderV14 from Stage 1 is loaded with pretrained weights and **frozen** (no gradients). This reuses the spatial hand-shape understanding and angle-primary features learned in Stage 1. After epoch 30, the encoder is **unfrozen** with a 0.1x learning rate multiplier for fine-tuning -- the encoder needs to slightly adapt to continuous input where signs have different dynamics than isolated clips.

#### Clip Processing

The variable-length input is split into 32-frame clips. If the last chunk has fewer than 32 frames, it's zero-padded to 32. All clips from all batch items are concatenated into one large batch and processed through the encoder in a **single forward pass** for GPU efficiency. Each clip -> encoder -> `[32, 384]` frame embeddings.

#### MultiScaleTCN (Temporal Convolutional Network)

**What is a TCN?** A Temporal Convolutional Network uses 1D convolutions along the time axis to capture temporal patterns. Unlike RNNs, TCNs process all timesteps in parallel and have a fixed, controllable receptive field.

**What makes it "Multi-Scale"?** Three parallel depthwise convolution branches with different kernel sizes capture patterns at different temporal scales simultaneously:

- **Kernel 3:** Fine-grained transitions (finger wiggles, rapid handshape changes)
- **Kernel 5:** Medium motion patterns (wrist arcs, hand rotations)
- **Kernel 9:** Broad stroke patterns (full sign movement trajectory)

Each branch: `Conv1d(d_model, d_model, k, depthwise) -> GroupNorm(8) -> GELU`

The three outputs are concatenated channel-wise (`384 x 3 = 1152`), fused through `Linear(1152 -> 384) -> LayerNorm -> GELU`, then **AdaptiveAvgPool1d(4)** collapses each 32-frame clip down to **4 tokens**.

**Why 4 tokens?** This is a compression ratio of 8x. Each sign (32 frames) becomes 4 semantic tokens. This is critical for CTC: having too many output tokens (32 per sign) would make the blank-dominated output hard to decode, while too few (1) would lose temporal information. 4 is empirically optimal -- enough to represent sign onset, stroke, peak, and offset.

#### SequenceTransformer (4 layers)

The per-clip 4 tokens are concatenated across all clips, giving a sequence of `num_clips x 4` tokens. This variable-length sequence is fed to a **4-layer Transformer encoder** that models relationships across signs:

```
Learned positional encoding (max_len=512) + input
-> 4x TransformerEncoderLayer(d=384, 8 heads, FFN=1536, GELU, pre-norm)
-> LayerNorm -> output [B, num_clips*4, 384]
```

**Padding mask** ensures attention doesn't attend to padding positions (sequences in a batch have different lengths).

**Key design choice:** This uses `nn.ModuleList` of `TransformerEncoderLayer`, NOT `nn.TransformerEncoder`. This matters for checkpoint compatibility -- the state dict keys are different (`layers.0.` vs `encoder.layers.0.`).

#### CTC Head

`Linear(384, 311)` maps each token to 311 logits: 310 sign classes + 1 blank token.

### CTC (Connectionist Temporal Classification)

**CTC** (Graves et al. 2006) is an alignment-free loss function designed for sequence-to-sequence problems where you don't know the exact alignment between input and output.

**The core problem:** Given a 5-sign sentence producing 20 output tokens (5 clips x 4 tokens each), you know the output should be "HELLO HOW ARE YOU DOING" but you don't know which of the 20 tokens correspond to which signs. CTC solves this.

**How CTC works:**

1. **The blank token (index 0):** CTC introduces a special "blank" symbol. The network can output blank at any position, meaning "no sign is being produced here" (transition between signs, preparation phase, etc.).

2. **Many-to-one mapping:** CTC defines a collapsing function: remove all blanks, then merge consecutive identical tokens. So `[blank, HELLO, HELLO, blank, blank, HOW, blank, ...]` collapses to `[HELLO, HOW, ...]`.

3. **The loss:** For a given target sequence, CTC sums the probabilities of ALL possible alignments that collapse to that target (using dynamic programming, similar to forward-backward in HMMs). The loss is `-log(sum of all valid alignment probabilities)`.

4. **Decoding:** At inference, **greedy decoding** takes `argmax` at each position, then collapses. **Beam search** considers multiple hypotheses simultaneously for better results.

**Critical constraints:**
- `blank = index 0` (MUST be first in vocabulary) and `PAD != blank` -- confusing these causes alignment errors
- Input length must be >= target length (you need at least as many output tokens as signs in the sentence)

#### InterCTC (Intermediate CTC)

An intermediate CTC head (`inter_ctc_proj`) is applied after the first SequenceTransformer layer. This provides an auxiliary loss that:
1. Ensures early layers learn meaningful representations (not just passing information through)
2. Provides gradient signal directly to early layers (avoiding vanishing gradients through 4 transformer layers)

The intermediate logits are projected and a CTC loss is computed, weighted by `inter_ctc_weight=0.1`.

#### Focal CTC Loss

`focal_ctc_loss` (Feng 2019) downweights easy samples (where CTC assigns high probability to the correct alignment) and focuses on hard samples:

```python
pt = exp(-CTC_loss)
focal_weight = (1 - pt) ^ gamma
loss = focal_weight * CTC_loss
```

With `gamma=2.0`, easy samples (pt close to 1) get near-zero weight, while hard samples (pt close to 0) get full weight.

#### CR-CTC (Consistency Regularization CTC)

CR-CTC (Yao et al. ICLR 2025) feeds two differently-augmented views of the same input through the model. Beyond the standard CTC loss on view 1, a **bidirectional KL-divergence** loss enforces that both views produce consistent output distributions:

```python
kl_loss = (KL(probs1 || probs2) + KL(probs2 || probs1)) / 2
total_loss = ctc_loss + cr_ctc_weight * kl_loss
```

This regularizes the model to be robust to augmentation and prevents overfitting to specific augmentation patterns. `cr_ctc_weight=0.3`.

### Synthetic Dataset (SyntheticCTCDataset)

Since collecting real continuous signing data is expensive (the project has 780 real phrase videos), Stage 2 primarily trains on **synthetic sequences** -- isolated sign clips concatenated with realistic transitions:

1. **Sequence composition:** 1-8 signs randomly selected per sample (10,000 total):
   - 10% single-sign sequences (edge case)
   - 10% long sequences (7-8 signs)
   - 80% medium sequences (2-6 signs)

2. **Confused gloss upweighting:** Signs that Stage 1 frequently confuses are sampled 3x more often, creating harder training examples.

3. **Hold trimming:** 2 frames trimmed from clip start/end to remove static "hold" poses that exist in isolated signs but not in continuous signing. ASL signs follow a Hold-Movement-Hold model, but the holds at segment boundaries are artifacts of isolated collection.

4. **Segment boundary jitter:** +/- 3 frames random shift to simulate imperfect segmentation in real data.

5. **Speed perturbation:** 0.8-1.2x speed variation per individual sign (different signs within one sequence get different speeds).

6. **Prosodic lengthening:** The last sign in a sequence is stretched by 1.3x (signers naturally slow down at phrase boundaries -- this is a well-documented linguistic feature of ASL).

7. **Minimum-jerk transitions (the most important augmentation):** Between signs, realistic transition frames are synthesized using a **5th-order polynomial** (Flash & Hogan 1985):

   ```
   s(t) = 10*t^3 - 15*t^4 + 6*t^5
   ```

   This produces a **bell-shaped velocity profile** -- accelerate, peak, decelerate -- which is how human arms actually move between positions (not linear interpolation). Duration scales with hand displacement (**Fitts' Law**): bigger movements get more transition frames (4-14 frames). Kinematics (velocity, acceleration) are recomputed from the interpolated XYZ.

8. **Random pauses:** 10% chance of inserting a 2-5 frame hold between signs (simulating hesitation or processing pauses).

9. **Temporal drop:** Randomly dropping 15% of frames after concatenation (simulates dropped frames / variable framerate).

10. **Gaussian jitter:** Adding small noise (std=0.01) to XYZ landmarks (simulates tracker noise).

11. **Post-processing:** After concatenation, the full XYZ sequence is smoothed with a moving average (window=5) to reduce concatenation artifacts at stitch boundaries. Then kinematics are recomputed ONCE from the smoothed XYZ, and bone features are appended.

**Why "warp XYZ first, recompute kinematics"?** This is an inviolable constraint. If you apply temporal warping to the full 16-channel tensor (which includes pre-computed velocity, acceleration, and bone features), the derivative channels become corrupted -- they no longer represent the actual derivatives of the warped XYZ. Always warp XYZ first, then recompute velocity, acceleration, and bone features from the warped positions.

### Real Phrase Dataset (RealPhraseCTCDataset)

The project also has 780 real continuous signing videos (9 phrases like "GOOD MORNING", "THANK YOU HOW YOU") that are mixed into training at 3x oversampling weight. These provide the model with real coarticulation patterns that synthetic data can only approximate.

### Online Augmentation (batch-level)

Applied on GPU during training:
- **Random 3D rotation** (up to 10 degrees) -- applied to XYZ, velocity, and acceleration channels via rotation matrices
- **Random scale** (0.85-1.15x) -- simulates different distances from camera
- **Gaussian noise** (std=0.003) on all spatial channels

### Training Details

- **Optimizer:** AdamW, lr=5e-4 with cosine warmup (5 epochs warmup)
- **Batch size:** 32
- **Epochs:** 60, patience=35
- **Encoder unfreezing:** Epoch 30, with 0.1x learning rate
- **Mixed precision (AMP):** FP16 on GPU for speed
- **EMA:** decay=0.999

### Output

CTC log-probabilities -> greedy/beam decoding -> gloss sequence (e.g., "HELLO HOW YOU").

**Stage 2 achieves 5.17% WER** (Word Error Rate) on synthetic validation data.

---

## Stage 3 -- Translation (Gloss -> English)

**Goal:** Translate an ASL gloss sequence into natural, conversational English.

**Core file:** `src/train_stage_3.py`

### Why This Stage Exists

ASL glosses are NOT English. ASL has its own grammar:
- "NAME WHAT YOU" -> "What is your name?"
- "YESTERDAY I BUY FOOD" -> "I bought food yesterday."
- "TOMORROW WE GO SCHOOL" -> "We're going to school tomorrow."

ASL lacks articles (a, the), copulas (is, are), and often uses topic-comment order. Stage 3 must learn this grammatical transformation.

### Architecture: Flan-T5-Base (250M parameters)

**T5** (Text-to-Text Transfer Transformer) is an encoder-decoder model: the encoder reads the input (gloss sequence), and the decoder generates the output (English) token by token.

**Flan-T5** is a variant fine-tuned on 1,800+ tasks with instruction-following formatting. This means it responds well to prompts like "Translate this ASL gloss to natural conversational English: HELLO HOW YOU".

### Input Format

Prompts follow a specific format:

- **Without context:** `"Translate this ASL gloss to natural conversational English: HELLO HOW YOU"`
- **With dialogue context:** `"[Previous: A: Hello, how are you?] Translate this ASL gloss to natural conversational English: I GOOD THANK-YOU"`

Context supports up to 4 previous dialogue turns, enabling the model to disambiguate based on conversation history (e.g., "FINE" in response to "How are you?" vs "How's the weather?").

### Noisy Gloss Augmentation

Stage 2's CTC output isn't perfect -- it makes characteristic errors. Stage 3 is trained to be robust to these by augmenting 30% of training examples with simulated CTC errors:

| Error Type | Weight | Simulation |
|---|---|---|
| **Deletion** | 40% | Remove a random gloss (CTC merges similar signs) |
| **Substitution** | 30% | Replace a gloss with a random one (Stage 1 confusion) |
| **Insertion** | 15% | Insert a random gloss (CTC spurious activation) |
| **Repetition** | 15% | Duplicate a gloss (CTC stutter on confident frames) |

Longer sequences get 2-3 errors; shorter ones get 1. The target English stays the same -- the model learns that "I GOOD HELP THANK-YOU" (with spurious HELP) should still translate to "I'm good, thank you."

### Training Configuration

- **Model:** `google/flan-t5-base` (250M params, encoder-decoder)
- **Max input length:** 96 tokens
- **Max target length:** 64 tokens
- **Batch size:** 32, gradient accumulation=2 (effective 64)
- **Learning rate:** 2e-4, cosine schedule with 200-step warmup
- **Epochs:** 25 with early stopping (patience=7)
- **Label smoothing:** 0.1
- **FP16:** Enabled on GPU
- **Beam search:** 4 beams at inference (considers 4 hypotheses simultaneously)
- **Metrics:** BLEU (n-gram overlap) and ROUGE-L (longest common subsequence)

### Loss Function

Standard **seq2seq cross-entropy** with label smoothing=0.1: for each output token position, the loss is cross-entropy between the model's predicted distribution and the (smoothed) target token.

### Dataset

~28,333 gloss-to-English pairs generated from the 310-class vocabulary, including:
- Single words: "HELLO" -> "Hello!"
- Phrases: "I GO STORE" -> "I'm going to the store."
- With dialogue context: "[Previous: A: Hello, how are you?] I GOOD THANK-YOU" -> "I'm good, thank you."

Plus noisy augmented copies (effectively doubling the dataset).

### Output

Natural English text. Example translations:

| Input Gloss | Context | Output |
|---|---|---|
| HELLO | - | Hello! |
| I GO STORE | - | I'm going to the store. |
| NAME WHAT YOU | - | What's your name? |
| I GOOD THANK-YOU | "A: Hello, how are you?" | I'm good, thank you. |

### Checkpoint

The correct checkpoint is `weights/slt_final_t5_model/` -- this one translates properly (an earlier checkpoint saved as `slt_conversational_t5_model` was less accurate).

---

## Test-Time Augmentation (TTA)

**TTA** is a technique used at inference to improve accuracy by averaging predictions across multiple augmented versions of the input.

### Mirror Averaging

For each input clip, the pipeline runs inference twice:
1. **Original** -- as-is
2. **Mirrored** -- swap left hand (nodes 0-20) with right hand (nodes 21-41), and flip the X coordinate

The softmax probabilities from both runs are averaged. This exploits the symmetry of many ASL signs -- a right-handed signer producing "HELLO" should get the same prediction as a left-handed signer.

### Mirror TTA Implementation (`_mirror_tta`)

The mirror operation flips these specific channels:
- X coordinate (ch 0)
- X velocity (ch 3)
- X acceleration (ch 6)
- Bone direction X (ch 10)
- Bone motion X (ch 13)

It does **NOT** flip the mask channel (ch 9) -- that would corrupt hand presence flags (changing "left hand present" to "left hand absent").

### Where TTA is Applied

- `docker/run_inference.py`: Stage 1 + Stage 2 + sliding window
- `src/camera_inference.py`: Stage 2 continuous recognition

---

## Glossary of Key Terms

| Term | Definition |
|---|---|
| **DS-GCN** | Depthwise Separable Graph Convolutional Network. Splits graph convolution into depthwise (per-partition spatial aggregation) and pointwise (channel mixing) operations. More efficient than full graph convolution. |
| **CTC** | Connectionist Temporal Classification. Alignment-free loss function for sequence-to-sequence problems. Introduces a blank token and marginalizes over all possible alignments via dynamic programming. |
| **ArcFace** | Angular margin loss from face recognition (Deng et al. 2019). Adds a fixed angular margin to the correct class's cosine similarity, forcing tighter intra-class clustering and wider inter-class separation. |
| **MultiScaleTCN** | Multi-scale Temporal Convolutional Network. Parallel 1D convolution branches with different kernel sizes (3, 5, 9) capture temporal patterns at different scales, followed by fusion and pooling. |
| **SequenceTransformer** | A Transformer encoder (using nn.ModuleList) that models temporal relationships across CTC tokens from multiple clips. Handles variable-length input with padding masks. |
| **One-Euro Filter** | Adaptive low-pass filter (Casiez et al. 2012). Smoothing strength adapts to signal speed: heavy smoothing for slow/still signals (jitter removal), light smoothing for fast signals (preserves dynamics). |
| **Z Depth Estimation** | Relative depth estimated from 2D keypoints using perspective projection. Anatomically rigid structures (palm length, inter-ear distance) have fixed real-world sizes, so their apparent 2D size inversely correlates with distance from camera. `Z = reference_size / observed_size`. |
| **Bone Features** | Per-node vectors computed as `xyz[child] - xyz[parent]` along skeleton edges. Captures hand shape independently of position. Bone motion is the temporal derivative of bone direction. Computed during extraction and stored in the .npy files (channels 10-15). |
| **Geo Features** | 114 geometric measurements computed from XYZ at runtime: fingertip distances, curl ratios, joint angles, palm normals, hand-to-face distances, etc. Provides explicit, interpretable features to complement learned GCN features. |
| **Temporal Resampling** | Interpolating a variable-length sequence to a fixed number of frames (32) using linear interpolation. Ensures consistent input dimensions regardless of video length/framerate. |
| **TTA** | Test-Time Augmentation. Running inference on multiple augmented versions of the input and averaging predictions. Mirror TTA swaps hands + flips X to exploit left-right symmetry. |
| **Coarticulation** | The blending of adjacent signs in continuous signing. The end of one sign overlaps with the beginning of the next, creating transition frames that don't match any isolated sign. Analogous to connected speech in spoken language. |
| **Minimum-Jerk Trajectory** | A 5th-order polynomial `s(t) = 10t^3 - 15t^4 + 6t^5` that produces bell-shaped velocity profiles (Flash & Hogan 1985). Models how humans actually move their limbs between positions -- accelerate smoothly, peak, decelerate smoothly. |
| **Fitts' Law** | The time to move to a target is proportional to the distance and inversely proportional to target size. Used to determine transition frame duration -- bigger hand movements get more transition frames. |
| **EMA** | Exponential Moving Average of model weights. Maintains `shadow_weight = decay * shadow_weight + (1 - decay) * current_weight` at each step. Used for evaluation -- smoother than raw training weights. |
| **DropPath / Stochastic Depth** | Randomly skipping entire layers during training. Linearly increasing drop probability (0% at first layer, 10% at last) regularizes deep networks. |
| **Drop-Graph** | Randomly masking graph nodes during training (5% probability). Forces the network to not over-rely on any single joint, improving robustness to partial occlusion. |
| **Savitzky-Golay Filter** | Fits a local polynomial to a sliding window of data points, then differentiates the polynomial analytically. Produces smoother derivatives than finite differences on noisy data. |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization. Normalizes local contrast in images, making pose estimation more robust to uneven lighting. |
| **CR-CTC** | Consistency Regularization CTC (Yao et al. ICLR 2025). Feeds two augmented views through the model and enforces KL-divergence consistency between their output distributions. |
| **Focal Loss** | Downweights easy examples and focuses on hard ones: `weight = (1 - pt)^gamma`. Applied to CTC in Stage 2 (gamma=2.0). Disabled in Stage 1 (gamma=0.0). |
| **Label Smoothing** | Softens hard one-hot targets by mixing with uniform distribution: `target = (1-eps)*one_hot + eps/K`. Prevents overconfidence and improves generalization. |
| **Prosodic Lengthening** | The linguistic phenomenon where the last sign in a phrase is produced more slowly. Simulated in synthetic data by stretching the final sign to 1.3x duration. |

---

## End-to-End Data Flow Example

Tracing "HELLO HOW ARE YOU" through the full pipeline:

1. **Video recorded** on webcam (2 seconds, 60fps, 120 frames)

2. **Stage 0 - Extraction:**
   - RTMW-XL detects 133 keypoints per frame -> select 61 (hands + face + body)
   - Temporal coherence rejection removes 3 frames with false wrist jumps
   - Z depth estimated via perspective projection (palm length ratio to sequence median)
   - 1-Euro filter smooths jitter while preserving fast signing motion
   - Linear interpolation fills 12 missing left-hand frames
   - Bone lengths stabilized to median across sequence
   - Centered on median wrist position, scaled by palm length
   - Resampled to 32 frames per sign (4 signs x 32 = 128 frames total for continuous input, or 32 per isolated clip)
   - Kinematics: velocity (Savitzky-Golay 1st derivative), acceleration (2nd derivative)
   - Mask: 1.0 for detected body parts, 0.0 for absent
   - Output: `[128, 61, 16]` float16 (continuous) or 4 separate `[32, 61, 16]` (isolated)

3. **Stage 1 - Classification (if isolated clips):**
   - Load `[32, 61, 16]` (all channels pre-computed)
   - DSGCNEncoderV14: 4 GCN blocks with SE attention (spatial hand-shape features) -> node attention -> concat 118 angle features -> 4 dilated TCN blocks (temporal relationships)
   - ArcFace head: frame attention pooling -> angular margin classification -> 310-class softmax
   - Prediction: "HELLO" (91.74% accuracy)

4. **Stage 2 - Continuous Recognition:**
   - Input: `[1, 128, 61, 16]` (128 frames = 4 clips of 32)
   - Split into 4 clips of `[32, 61, 16]` (all 16 channels already present)
   - Frozen Stage 1 DSGCNEncoderV14 -> `[4, 32, 384]` per-clip embeddings
   - MultiScaleTCN -> `[4, 4, 384]` (32 frames -> 4 tokens per clip)
   - Reshape to `[1, 16, 384]` (4 clips x 4 tokens = 16 sequence tokens)
   - SequenceTransformer (4 layers) -> `[1, 16, 384]`
   - CTC head -> `[1, 16, 311]` logits
   - CTC greedy decode: `[blank, HELLO, HELLO, blank, HOW, blank, blank, ARE, blank, YOU, YOU, blank, blank, blank, blank, blank]` -> collapse -> `[HELLO, HOW, ARE, YOU]`

5. **Stage 3 - Translation:**
   - Input: `"Translate this ASL gloss to natural conversational English: HELLO HOW ARE YOU"`
   - Flan-T5-Base encoder processes input tokens
   - Decoder generates: "Hello, how are you?"
   - Beam search (4 beams) selects highest-probability translation

6. **Final output:** "Hello, how are you?"
