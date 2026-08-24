# Addressing the Adviser's Request for Single End-to-End (E2E) Training

**The Question:** *Can we combine everything into 1 single training loop so that there will be only 1 chart and graph?*

While End-to-End (E2E) models are popular, for our specific architecture and dataset constraints, **switching to 1 single training phase is highly discouraged for this capstone.** 

Our current 3-stage modular pipeline is an engineering solution to a very hard problem. Here is a detailed breakdown of why we must keep the 3 stages, and the compromise we can offer to satisfy the adviser's request for a "single metric."

---

## 1. The Data Scarcity Problem (Why the modular approach is smart)
To train an End-to-End model, we would need a massive dataset of parallel paired data: **(Continuous ASL Video $\rightarrow$ Natural English Sentences)**. These datasets (like Phoenix-2014T) are extremely rare, small, and very hard to create. 

By breaking the system into 3 stages, we bypass this data limitation:
*   **Stage 1** uses isolated sign videos (which are relatively easy to find/record).
*   **Stage 2** chains isolated signs together to create *synthetic* continuous video data.
*   **Stage 3** uses *synthetically generated text* to map glosses to fluent English.

If combined into one training loop, we lose the ability to use our synthetic text data and isolated sign videos. We would need thousands of hours of real continuous ASL video labeled with fluent English translations just to make the model converge.

## 2. The Architecture & Gradient Flow Problem
In a single training loop, the "loss" calculated at the end (English text errors) must backpropagate all the way to the beginning (the video GCN). 
*   **The Disconnect:** Our Stage 3 model (T5) expects discrete text tokens (words) as inputs. Our Stage 2 model outputs probabilities (via CTC) over a timeline. 
*   We cannot easily pass gradients back through discrete text tokens (argmax) into a CTC decoder and then into a GCN without highly advanced techniques (like Gumbel-Softmax or Reinforcement Learning). Building that bridging layer is a Ph.D.-level research problem on its own.

## 3. The "Black Box" Debugging Problem
With a 3-stage pipeline, if the translation is wrong, we know exactly why:
*   Did it fail to see the hand shape? (Stage 1 error)
*   Did it miss a word in the sequence? (Stage 2 CTC error)
*   Did it conjugate the verb wrong? (Stage 3 T5 error)

In an E2E model, it becomes a total black box. If it outputs the wrong sentence, we won't know if the computer vision failed or if the NLP model hallucinated. 

---

## The Solution: End-to-End *Evaluation* (Not Training)

The adviser's core desire is likely to see **how the system performs as a whole**, rather than looking at three disjointed pieces. We don't need End-to-End *Training* to provide an End-to-End *Evaluation*.

**Our Proposal:**
1.  **Keep the 3 training phases separate** (because of the data/architecture limitations above).
2.  **Create an End-to-End Testing Script** (`evaluate_pipeline.py`) that feeds a test set of raw continuous videos into the completely combined pipeline: `Video -> Extract -> DS-GCN -> BiLSTM/CTC (Gloss) -> T5 -> Final English Text`.
3.  **Present ONE Unified Final Graph/Metric.** Evaluate the final predicted English text against the ground-truth human English text using a standard NLP metric like **BLEU-4** or **ROUGE** scores. 

### What to say to the adviser:
> *"We investigated an end-to-end training approach, but due to the architectural gap between CTC continuous decoding and the T5 LLM token space, as well as a lack of massive parallel Video-to-English datasets, end-to-end backpropagation isn't feasible. Our modular 3-stage pipeline acts as a bridge, allowing us to leverage synthetic data to solve the data scarcity problem. However, to evaluate the system holistically, we will run an end-to-end inference pass and provide a unified final metric (BLEU score) to demonstrate the complete pipeline's overall accuracy."*