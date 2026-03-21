# Comprehensive System Analysis Prompt

*Instructions: Copy the text below the line and paste it into your AI assistant along with your entire codebase (Extract, Stage 1, Stage 2, Stage 3, and Inference scripts) to trigger a deep architectural review.*

---

<SYSTEM_ROLE>
You are an elite AI Architect and Machine Learning Engineer specializing in Computer Vision, Graph Neural Networks (GCNs), and Sequence-to-Sequence NLP pipelines.
</SYSTEM_ROLE>

<OBJECTIVE>
Conduct a deep, comprehensive architectural and code-level review of my 3-Stage Sign Language Translation (SLT) system. Your goal is to maximize the final End-to-End translation accuracy (BLEU/WER scores) by identifying bottlenecks, data leaks, train/test mismatches, and mathematical inefficiencies.
</OBJECTIVE>

<INSTRUCTIONS>
1. **System Mapping:** Briefly map the data flow from Raw Video -> Extraction (Landmarks) -> Stage 1 (DS-GCN) -> Stage 2 (BiLSTM/CTC) -> Stage 3 (T5) -> English Text.
2. **Mental Simulation & Validation:** When you propose a recommendation, mentally simulate "trying" it within the constraints of Kaggle T4/P100 hardware and my current dataset size. If your simulated solution would cause OOM errors, dimension mismatches, or massive training slowdowns, discard it and immediately propose a more robust alternative.
3. **Codebase Check:** Audit the provided code for any hidden bugs, gradient flow issues, or tensor shape mismatches.
4. **No Code Modification:** Provide your findings as a detailed report with code *snippets* for suggestions. Do not rewrite my entire codebase.

<SPECIFIC_SCENARIOS_TO_ANALYZE>
Please provide a deep theoretical and practical analysis of the following "What-If" scenarios. For each, tell me if it will improve the final accuracy, hurt it, or break the architecture, and provide the best implementation strategy:

**What-If 1: Train/Test Distribution Matching (Missing Frames)**
*Scenario:* What if we ensure that during live inference, missing frames (failed MediaPipe detections) are filled in and interpolated *exactly* the same way they are during the `extract.py` training data generation? 
*Question:* How much does a mismatch here hurt accuracy, and how exactly should the inference pipeline be structured to guarantee identical preprocessing?

**What-If 2: Variable Temporal Resolution (Removing the 32-Frame Limit)**
*Scenario:* What if we *stop* forcing every video clip into exactly 32 frames? What if we read all frames, do not skip frames, do not downscale/resize the video, and just let the raw sequence length pass through the network?
*Question:* How would the DS-GCN and Transformer Encoder handle variable-length temporal sequences? What changes would be required for padding, packing (`pack_padded_sequence`), and masking to prevent the model from collapsing? Evaluate the tradeoff between the loss of temporal data (resampling to 32) versus the computational/VRAM explosion of processing 200+ frames per video.

**What-If 3: Signer Speed Augmentation**
*Scenario:* Signers sign at different speeds (fast native signers vs. slow beginners). 
*Question:* How can we implement temporal augmentations (like dynamic Time Warping or random frame dropping) to make the model robust to signer speed? Provide a mathematical/code implementation of this augmentation that works on my `[B, T, 42, 10]` tensor without breaking the kinematics (velocity/acceleration) calculations.
</SPECIFIC_SCENARIOS_TO_ANALYZE>

<DELIVERABLES>
Provide a highly structured Markdown report containing:
1. **System Health Check:** Critical bugs or train/test mismatches found.
2. **Scenario Analysis:** Detailed answers to the 3 "What-Ifs" above, with simulated outcomes.
3. **Top 3 Accuracy Recommendations:** The three highest-impact changes I can make *right now* to improve my final BLEU/WER scores, prioritizing changes that require minimal architectural rewrites.
4. **Implementation Snippets:** Python code blocks demonstrating how to implement your best recommendations.
</DELIVERABLES>

Take a deep breath, think step-by-step through the tensor shapes and gradient flows, and begin your analysis.