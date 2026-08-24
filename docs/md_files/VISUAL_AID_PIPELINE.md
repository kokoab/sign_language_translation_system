# SLT Pipeline Visual Aid — Full Architecture Diagram

Use this as a reference to recreate in PowerPoint, Canva, or draw on a poster.

---

## MASTER DIAGRAM: Full Pipeline Flow

```
===================================================================================
                          COMPLETE SLT PIPELINE
===================================================================================

  WEBCAM VIDEO                                              ENGLISH TEXT
  (raw frames)                                              "I'm going to
       |                                                     the store."
       v                                                         ^
+============+     +============+     +============+     +============+
||            ||     ||            ||     ||            ||     ||            ||
||  STAGE 0   ||---->||  STAGE 1   ||---->||  STAGE 2   ||---->||  STAGE 3   ||
|| Extraction ||     ||  Classify  ||     || Continuous ||     || Translate  ||
||            ||     ||            ||     ||            ||     ||            ||
+============+     +============+     +============+     +============+
  RTMW-XL          DS-GCN-TCN          CTC + TCN +         Flan-T5
  61 nodes          + ArcFace          Transformer          250M params
  16 channels        5.4M params        8.0M params

  [32,61,16]         310 classes        Gloss sequence      Natural English
                     91.74% acc         10.96% WER          BLEU 72.90
```

---

## STAGE 0: Extraction (Video -> Tensor)

```
+------------------+
|   WEBCAM VIDEO   |
|  (RGB frames)    |
+--------+---------+
         |
         v
+------------------+
| IMAGE PREPROCESS |
| - Gamma correct  |
| - CLAHE          |
| - Bilateral      |
| - Unsharp mask   |
+--------+---------+
         |
         v
+------------------+
|    RTMW-XL       |
| Pose Estimation  |
| 133 keypoints    |
| per frame        |
+--------+---------+
         |
         v
+------------------+
| SELECT 61 NODES  |
|                  |
| 0-20:  L Hand    |
| 21-41: R Hand    |
| 42-56: Face (15) |
| 57-60: Body (4)  |
+--------+---------+
         |
         v
+------------------+
| NORMALIZE TO     |
| [0, 1] RANGE     |
| X = px / width   |
| Y = px / height  |
| Z = estimated    |
+--------+---------+
         |
         v
+-------------------------------+
| SIGNAL PROCESSING             |
|                               |
| 1. Z Depth Estimation         |
|    (palm length ratio)        |
|                               |
| 2. Temporal Coherence         |
|    Rejection (remove jumps)   |
|                               |
| 3. 1-Euro Adaptive Filter     |
|    (smooth jitter, keep fast) |
|                               |
| 4. Linear Interpolation       |
|    (fill missing frames)      |
|                               |
| 5. Bone Length Stabilization   |
|    (median normalization)     |
|                               |
| 6. Centering & Scaling        |
|    (translation/scale invar.) |
|                               |
| 7. Temporal Resampling        |
|    (any length -> 32 frames)  |
+-------------------------------+
         |
         v
+-------------------------------+
| COMPUTE 16 CHANNELS           |
|                               |
| Ch 0-2:   XYZ position       |
| Ch 3-5:   Velocity (dx,dy,dz)|
| Ch 6-8:   Acceleration       |
| Ch 9:     Detection mask     |
| Ch 10-12: Bone direction     |
| Ch 13-15: Bone motion        |
+-------------------------------+
         |
         v
   +-------------+
   | OUTPUT:     |
   | [32, 61, 16]|
   | .npy file   |
   +-------------+
```

---

## STAGE 1: Isolated Sign Classification

```
   +------------------+
   | INPUT:           |
   | [B, 32, 61, 16]  |
   +--------+---------+
            |
            v
+===========================+
|     DS-GCN ENCODER        |
|                           |
|  +---------------------+  |
|  | input_proj           |  |
|  | Linear(16 -> 96)     |  |
|  | LayerNorm + GELU     |  |
|  +----------+----------+  |
|             |              |
|             v              |
|  +---------------------+  |
|  | GCN Block 1          |  |
|  | 96 -> 192, kernel=3  |  |
|  | DS spatial + temporal |  |
|  +----------+----------+  |
|             |              |
|             v              |
|  +---------------------+  |
|  | GCN Block 2          |  |
|  | 192 -> 384, kernel=5 |  |
|  | DS spatial + temporal |  |
|  +----------+----------+  |
|             |              |
|             v              |
|  +---------------------+  |
|  | GCN Block 3 + SE     |  |
|  | 384 -> 384, kernel=5 |  |
|  | DS spatial + temporal |  |
|  | + channel recalib.   |  |
|  +----------+----------+  |
|             |              |
|             v              |
|  +---------------------+  |
|  | GCN Block 4 + SE     |  |
|  | 384 -> 384, kernel=7 |  |
|  | DS spatial + temporal |  |
|  | + channel recalib.   |  |
|  +----------+----------+  |
|             |              |
|             v              |
|  +---------------------+  |
|  | NODE ATTENTION       |  |
|  | softmax over 61      |  |
|  | nodes -> weighted    |  |
|  | sum                  |  |
|  | [B,32,61,384]        |  |
|  |    -> [B,32,384]     |  |
|  +----------+----------+  |
+===========================+
              |
    +---------+---------+
    |                   |
    v                   v
+----------+   +-----------------+
| GCN out  |   | ANGLE FEATURES  |
| [B,32,   |   | 59 angles +     |
|  384]    |   | 59 velocities   |
+----+-----+   | = 118 features  |
     |         | [B, 32, 118]    |
     |         +--------+--------+
     |                  |
     +--------+---------+
              |
              v
     +------------------+
     | CONCATENATE      |
     | [B, 32, 502]     |
     | (384 + 118)      |
     +--------+---------+
              |
              v
     +------------------+
     | angle_proj       |
     | Linear(502, 384) |
     | [B, 32, 384]     |
     +--------+---------+
              |
              v
+===========================+
|     TEMPORAL TCN          |
|                           |
|  Block 1: dilation=1     |
|    sees 3 frames          |
|  Block 2: dilation=2     |
|    sees 5 frames          |
|  Block 3: dilation=4     |
|    sees 9 frames          |
|  Block 4: dilation=8     |
|    sees 17 frames         |
|                           |
|  [B, 32, 384]             |
+============+==============+
             |
             v
+===========================+
|     ARCFACE HEAD          |
|                           |
|  Frame Attention:         |
|    softmax over 32 frames |
|    [B,32,384] -> [B,384]  |
|                           |
|  L2 Normalize features    |
|  L2 Normalize 310         |
|    class templates        |
|  Dot product -> cosine    |
|  Add margin to correct    |
|    class (training only)  |
|  Scale by s=30            |
|                           |
|  Output: 310 class scores |
+===========================+
             |
             v
      +--------------+
      | PREDICTION:  |
      | "HELLO"      |
      | (1 of 310)   |
      +--------------+
```

---

## INSIDE A DS-GCN BLOCK (Detail)

```
    Input: [B, T, 61, C_in]
              |
   +----------+----------+
   |          |          |
   v          v          v
+------+  +------+  +------+
|A_self|  |A_out |  |A_in  |
|@ x   |  |@ x   |  |@ x   |
|* w1  |  |* w2  |  |* w3  |    <- cheap vector scaling (depthwise)
+--+---+  +--+---+  +--+---+
   |         |         |
   +----+----+----+----+
        |         |
        v         v
   +---------+----------+
   | CONCATENATE         |
   | [B, T, 61, 3*C_in] |
   +---------+-----------+
             |
             v
   +-------------------+
   | Linear(3*C, C_out) |    <- one matrix multiply (pointwise)
   +--------+----------+
            |
            v
   +-------------------+
   | Temporal Conv1D    |    <- along time axis
   | + GroupNorm         |
   +--------+----------+
            |
            v
   +-------------------+
   | + Residual         |
   | GELU + LayerNorm   |
   +-------------------+
            |
            v
    Output: [B, T, 61, C_out]
```

---

## STAGE 2: Continuous Sign Recognition (CTC)

```
   +---------------------------+
   | INPUT: Variable length    |
   | [B, T, 61, 16]            |
   | e.g., T=160 (5 signs)     |
   +------------+--------------+
                |
                v
   +---------------------------+
   | SPLIT INTO 32-FRAME CLIPS |
   |                           |
   | Clip 1: frames 0-31       |
   | Clip 2: frames 32-63      |
   | Clip 3: frames 64-95      |
   | Clip 4: frames 96-127     |
   | Clip 5: frames 128-159    |
   +------------+--------------+
                |
                v
+================================+
| FROZEN STAGE 1 ENCODER         |
| (same DS-GCN + angle + TCN)   |
|                                |
| All 5 clips processed at once  |
| [5, 32, 61, 16] -> [5, 32, 384]|
|                                |
| Frozen epochs 1-30             |
| Unfrozen epoch 30+ (0.1x lr)  |
+===============+=================+
                |
                v
+================================+
| MULTISCALE TCN                 |
|                                |
|  +--------+ +--------+ +--------+
|  |kernel=3| |kernel=5| |kernel=9|   3 parallel branches
|  | fine   | | medium | | broad  |
|  +---+----+ +---+----+ +---+----+
|      |          |          |
|      +-----+----+-----+---+
|            |           |
|            v           v
|     +-------------------+
|     | Concatenate: 1152  |
|     | Linear(1152, 384)  |
|     | LayerNorm + GELU   |
|     +--------+----------+
|              |
|              v
|     +-------------------+
|     | AdaptiveAvgPool1d  |
|     | 32 frames -> 4    |
|     | tokens per clip    |
|     +-------------------+
|                                |
| Per clip: [32, 384] -> [4, 384]|
+===============+=================+
                |
                v
   +---------------------------+
   | RESHAPE TO SEQUENCE       |
   |                           |
   | 5 clips x 4 tokens =     |
   | 20 tokens [B, 20, 384]    |
   +------------+--------------+
                |
                v
+================================+
| SEQUENCE TRANSFORMER           |
| (4 layers)                     |
|                                |
| + Learned positional encoding  |
| Layer 1: self-attn + FFN       |
| Layer 2: self-attn + FFN       |
| Layer 3: self-attn + FFN       |
| Layer 4: self-attn + FFN       |
| + LayerNorm                    |
|                                |
| Every token attends to every   |
| other token (cross-clip)       |
|                                |
| + Padding mask for variable    |
|   length sequences             |
|                                |
| [B, 20, 384]                   |
+===============+=================+
                |
                v
   +---------------------------+
   | CTC HEAD                  |
   | Linear(384, 311)          |
   | 310 signs + 1 blank       |
   +------------+--------------+
                |
                v
   +---------------------------+
   | CTC DECODING              |
   |                           |
   | Raw: [blank, HELLO, HELLO,|
   |  blank, blank, HOW, blank,|
   |  ARE, blank, YOU, YOU,    |
   |  blank, blank, ...]       |
   |                           |
   | Collapse: remove blanks,  |
   | merge consecutive same    |
   |                           |
   | Result: [HELLO, HOW,      |
   |          ARE, YOU]         |
   +---------------------------+
                |
                v
        +---------------+
        | GLOSS OUTPUT: |
        | "HELLO HOW    |
        |  ARE YOU"     |
        +---------------+
```

---

## STAGE 3: Gloss-to-English Translation

```
        +--------------------+
        | GLOSS INPUT:       |
        | "HELLO HOW ARE YOU"|
        +---------+----------+
                  |
                  v
        +--------------------+
        | CREATE PROMPT:     |
        | "Translate this    |
        |  ASL gloss to      |
        |  natural English:  |
        |  HELLO HOW ARE YOU"|
        +---------+----------+
                  |
                  v
+=================================+
|         FLAN-T5-BASE            |
|         (250M params)           |
|                                 |
|  +---------------------------+  |
|  |        ENCODER            |  |
|  | Processes entire prompt   |  |
|  | at once using self-attn   |  |
|  | -> rich representation    |  |
|  | of gloss meaning          |  |
|  +-------------+-------------+  |
|                |                |
|                v                |
|  +---------------------------+  |
|  |        DECODER            |  |
|  | Generates one word at     |  |
|  | a time, left to right:    |  |
|  |                           |  |
|  | Step 1: <start>           |  |
|  |   cross-attn to encoder   |  |
|  |   -> "Hello"              |  |
|  |                           |  |
|  | Step 2: "Hello"           |  |
|  |   self-attn (prev words)  |  |
|  |   cross-attn to encoder   |  |
|  |   -> ","                  |  |
|  |                           |  |
|  | Step 3: "Hello,"          |  |
|  |   -> "how"                |  |
|  |                           |  |
|  | Step 4: "Hello, how"      |  |
|  |   -> "are"                |  |
|  |                           |  |
|  | Step 5: "Hello, how are"  |  |
|  |   -> "you"                |  |
|  |                           |  |
|  | Step 6: "Hello, how are   |  |
|  |          you"             |  |
|  |   -> "?"                  |  |
|  |                           |  |
|  | Step 7: -> <end>          |  |
|  +---------------------------+  |
|                                 |
|  Beam search (4 beams):         |
|  Explores 4 parallel paths,     |
|  picks best complete sentence   |
+=================================+
                  |
                  v
        +--------------------+
        | ENGLISH OUTPUT:    |
        | "Hello, how are    |
        |  you?"             |
        +--------------------+
```

---

## TRAINING COMPONENTS (Overview)

```
+================================================================+
|                     STAGE 1 TRAINING                           |
|                                                                |
|  Loss: ArcFace angular margin + label smoothing (0.05)         |
|  Augmentation: mixup, rotation, scale, noise, Drop-Graph      |
|  Optimizer: AdamW, lr=3e-4                                     |
|  Schedule: cosine decay (no warm restarts)                     |
|  ArcFace warmup: CE only (ep 1-10) -> ramp (11-30) -> full    |
|  EMA: decay=0.999                                              |
|  Epochs: 150, patience=25                                      |
|  Batch: 256 x 4 accum = 1024 effective                        |
|  Data: ~57,535 samples, 310 classes, 7 signers                |
+================================================================+

+================================================================+
|                     STAGE 2 TRAINING                           |
|                                                                |
|  Loss: Focal CTC + InterCTC (0.1) + CR-CTC (0.3)             |
|  Data: 10,000 synthetic + 780 real phrases (3x oversampled)   |
|  Synthetic augmentation:                                       |
|    - Hold trimming, boundary jitter, speed warp               |
|    - Minimum-jerk transitions (Flash & Hogan 1985)            |
|    - Temporal drop, Gaussian jitter, smoothing                |
|  Online augmentation: rotation, scale, noise                   |
|  Optimizer: AdamW, lr=5e-4                                     |
|  Schedule: cosine warmup (5 epochs)                            |
|  Encoder: frozen 30 epochs, unfreeze at 0.1x lr               |
|  Epochs: 60, patience=35                                       |
|  Batch: 32                                                     |
+================================================================+

+================================================================+
|                     STAGE 3 TRAINING                           |
|                                                                |
|  Loss: seq2seq cross-entropy + label smoothing (0.1)          |
|  Data: ~28,333 gloss-English pairs + noisy augmented copies   |
|  Noisy gloss augmentation: simulate CTC errors                |
|    - 40% deletion, 30% substitution, 15% insertion, 15% repeat|
|  Model: google/flan-t5-base (250M, pretrained)                |
|  Optimizer: AdamW, lr=2e-4                                     |
|  Schedule: cosine with 200-step warmup                         |
|  Epochs: 25, early stopping patience=7                         |
|  Batch: 32 x 2 accum = 64 effective                          |
|  Beam search: 4 beams at inference                            |
+================================================================+
```

---

## DATA CLEANING PIPELINE

```
~67,000 raw videos
       |
       v
+------------------+
| RTMW-XL          |
| Extraction       |
| -> 66,451 .npy   |
+--------+---------+
         |
         v
+------------------+
| QUALITY FILTERS  |
| - All zeros?     |
| - No motion?     |
| - Spatial outlier?|
| Remove ~4-5k     |
+--------+---------+
         |
         v
+------------------+
| CONFIDENT        |
| LEARNING         |
| Run trained model|
| on all samples   |
| Remove bottom 3% |
| (~1.7-2k)        |
+--------+---------+
         |
         v
+------------------+
| CLEAN DATASET    |
| ~57,535 samples  |
| 310 classes      |
| 7 signers        |
+------------------+
```

---

## 61-NODE SKELETON MAP

```
              FOREHEAD (44)
                  |
    L_EYE (55)---NOSE (42)---R_EYE (56)
                  |
   L_EAR (45)  CHIN (43)  R_EAR (46)
                  |
  L_BROW(51,52)     R_BROW(53,54)
                  |
         UPPER_LIP (49)
    L_MOUTH(47)---+---R_MOUTH(48)
         LOWER_LIP (50)


   L_SHOULDER (57)--------R_SHOULDER (58)
        |                       |
   L_ELBOW (59)            R_ELBOW (60)
        |                       |
   L_WRIST (0)             R_WRIST (21)
      / | \                   / | \
     /  |  \                 /  |  \
  THUMB INDEX MIDDLE      THUMB INDEX MIDDLE
  1-4   5-8  9-12         22-25 26-29 30-33
        RING  PINKY              RING  PINKY
        13-16 17-20              34-37 38-41
```

---

## 16 INPUT CHANNELS

```
+-----+-----------------------------+---------------------------+
| Ch  | Name                        | What it captures          |
+-----+-----------------------------+---------------------------+
|  0  | X position                  | Left-right in frame       |
|  1  | Y position                  | Up-down in frame          |
|  2  | Z position (estimated)      | Relative depth            |
|  3  | X velocity                  | Horizontal speed          |
|  4  | Y velocity                  | Vertical speed            |
|  5  | Z velocity                  | Depth speed               |
|  6  | X acceleration              | Horizontal accel          |
|  7  | Y acceleration              | Vertical accel            |
|  8  | Z acceleration              | Depth accel               |
|  9  | Detection mask              | 1=detected, 0=absent      |
| 10  | Bone direction X            | Bone vector horizontal    |
| 11  | Bone direction Y            | Bone vector vertical      |
| 12  | Bone direction Z            | Bone vector depth         |
| 13  | Bone motion X               | Bone change horizontal    |
| 14  | Bone motion Y               | Bone change vertical      |
| 15  | Bone motion Z               | Bone change depth         |
+-----+-----------------------------+---------------------------+
```

---

## PARAMETER COUNT SUMMARY

```
+------------------+----------------+----------+
| Component        | Parameters     | % Total  |
+------------------+----------------+----------+
| Stage 1          |                |          |
|   DS-GCN Encoder |   ~4.0M       |   1.5%   |
|   TCN            |   ~1.0M       |   0.4%   |
|   ArcFace Head   |   ~0.4M       |   0.2%   |
|   Subtotal       |    5.4M       |   2.1%   |
+------------------+----------------+----------+
| Stage 2          |                |          |
|   Encoder(shared)|   (from S1)   |    --    |
|   MultiScaleTCN  |   ~2.0M       |   0.8%   |
|   SeqTransformer |   ~5.5M       |   2.1%   |
|   CTC Head       |   ~0.5M       |   0.2%   |
|   Subtotal (new) |    8.0M       |   3.1%   |
+------------------+----------------+----------+
| Stage 3          |                |          |
|   Flan-T5-Base   |  248.0M       |  94.8%   |
+------------------+----------------+----------+
| TOTAL            |  261.4M       | 100.0%   |
+------------------+----------------+----------+

Custom-built (Stages 1+2): 13.4M  -- lighter than ResNet-50 (25M)
Pretrained (Stage 3):     248.0M  -- fine-tuned, not built from scratch
```
