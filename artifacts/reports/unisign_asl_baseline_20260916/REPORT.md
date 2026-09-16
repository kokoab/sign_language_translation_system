# Released Uni-Sign ASL baseline

Completed pretrained inference with strict checkpoint loading. No training or live-system changes.

**Reviewed decision: do not replace Stage 2 or the Reel with this released checkpoint.**
All 21 predictions are present, and BLEU/chrF were independently recomputed from
the saved English references. Some validation outputs capture the topic, but others
change the meaning or invent details. For example, “I really like that” becomes
“And, I like that from a spoon.” The previously annotated HELLO HOW YOU clip becomes
a sentence about taking a brush, while the webcam EAT diagnostic yields “I'm sorry,
I'm sorry.” Fluent text therefore does not establish corrected recognition or
repetition. The webcam clips lack expert English references, so no numerical webcam
accuracy is claimed.

This rejects this checkpoint as a drop-in solution under the tested native-demo
pipeline; it does not reject gloss-free translation generally or isolate whether
pose quality, domain mismatch, or model learning dominates. No further training or
architecture change follows automatically. The remaining Stage-2 decision needs
verified target-domain examples distinguishing held signs from intentional repeats.

The deployment decision is not a rejection of fine-tuning. Some sentence predictions
preserve meaningful content: the background-painting example retains the activity
and background color, and the birds/flight example retains the topic and action.
Zero exact matches does not mean zero useful translations. This experiment did not
test adaptation. Its short-clip failures also confound task, signer, capture and
domain differences; they do not establish that isolated duration is the cause.
Uni-Sign supports isolated recognition through separate task fine-tuning, whereas
this checkpoint was evaluated as a How2Sign sentence translator. A potential
adaptation study would keep the native architecture and use verified target-domain
training pairs with separate unseen-signer evaluation; it has not been launched.

12 realigned How2Sign validation utterances: BLEU 8.21, chrF 38.06; exact text matches 0/12.

This is a small selected development slice, not a benchmark, a signer-disjoint claim, or evidence of improved webcam recognition. Automatic scores do not establish semantic correctness.

Checkpoint: 1.19 GB; 587,747,368 parameters. Native pose extraction on CPU; translation on mps float32.
Median pose extraction 11.19s and translation 0.91s per completed clip. Desktop timing only; no iPhone measurements.

## Paired validation outputs

### -d5dN54tH2E_0-1-rgb_front

Reference: We're going to work on a arm drill that will help you have graceful hand movements in front of you.

Prediction: The best way to do that is to actually move the body pretty much like you're dribbling.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_0-1-rgb_front.mp4)

### -d5dN54tH2E_1-1-rgb_front

Reference: I call it painting the wall.

Prediction: It's called a paint tool.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_1-1-rgb_front.mp4)

### -d5dN54tH2E_10-1-rgb_front

Reference: So we're going to go up and down; let's switch hands, down and up; down and up.

Prediction: So we're going to move one, two, three, four.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_10-1-rgb_front.mp4)

### -f1_kdl050s_0-1-rgb_front

Reference: In this clip we are going to talk about dangers for these birds in the household and otherwise.

Prediction: In this clip we're going to talk about how to get your birds dangerous within the house and the other things like that.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_0-1-rgb_front.mp4)

### -f1_kdl050s_1-1-rgb_front

Reference: The number one loss for these birds, is flight.

Prediction: The number one negative for birds is they're flying.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_1-1-rgb_front.mp4)

### -f1_kdl050s_10-1-rgb_front

Reference: You need to be very careful when cleaning their cages if these birds are flighted.

Prediction: You should be careful and clean them carefully for the other thing.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_10-1-rgb_front.mp4)

### 0pKzG0RRUz4_0-2-rgb_front

Reference: Another technique is you can braid the hair on first and then start wrapping cause some people have really short hair and we can create the look with adding the hair end.

Prediction: Another technique you can do to round off your hair, the first thing that you can do is start spraying and spraying because some people really youth hair when they can create a crease in the hair.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_0-2-rgb_front.mp4)

### 0pKzG0RRUz4_3-2-rgb_front

Reference: I really like that.

Prediction: And, I like that from a spoon.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_3-2-rgb_front.mp4)

### 0pKzG0RRUz4_4-2-rgb_front

Reference: You can get instant length, which I love.

Prediction: You can get a little bit of a tail that I love.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_4-2-rgb_front.mp4)

### 0zvsqf23tmw_1-2-rgb_front

Reference: And in this segment, I'm going to just start paint in the background color.

Prediction: In this series we're going to start by doing the painting of background color.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_1-2-rgb_front.mp4)

### 0zvsqf23tmw_10-2-rgb_front

Reference: I'm going to start on the edge so I can get the edge painted in against the skyline.

Prediction: We're going to start with the edge of the telescope and we're going to start with the shadow of the ceiling.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_10-2-rgb_front.mp4)

### 0zvsqf23tmw_11-2-rgb_front

Reference: I'm just doing a dark hill here.

Prediction: I'm not going to cut anything out of it.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_11-2-rgb_front.mp4)

## Earlier difficult recordings

These clips lack verified English translations. Gloss annotations are shown separately; no English accuracy is assigned. No count-correctness claim is made from fluent English.

### webcam_i

Prediction: He's going to go to the other side.

Previous CTC replay, phase zero: ['I']

Existing gloss annotation: None

Rows below are decoder windows, NOT sign annotations.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_i.mp4)

### webcam_hello

Prediction: I'm going to show you a little bit of what it looks like.

Previous CTC replay, phase zero: ['HELLO', 'HELLO']

Existing gloss annotation: None

Rows below are decoder windows, NOT sign annotations.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_hello.mp4)

### webcam_eat

Prediction: I'm sorry, I'm sorry.

Previous CTC replay, phase zero: ['I', 'EAT', 'EAT']

Existing gloss annotation: None

Rows below are decoder windows, NOT sign annotations.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_eat.mp4)

### asllrp_night_time

Prediction: We're going to start off with this one.

Previous CTC replay, phase zero: ['SCHOOL']

Existing gloss annotation: ['NIGHT', 'TIME']

Manifest marks reference intervals verified.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_night_time.mp4)

### asllrp_time_friend

Prediction: Do you have a good time?

Previous CTC replay, phase zero: ['FRIEND']

Existing gloss annotation: ['TIME', 'FRIEND']

Manifest does NOT mark intervals verified; no new expert annotation check.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_time_friend.mp4)

### asllrp_morning

Prediction: I guess in the bedroom, I guess it's easier to guess the hours.

Previous CTC replay, phase zero: ['DAY', 'TIME', 'TOMORROW', 'FAMILY', 'TOMORROW']

Existing gloss annotation: ['MORNING']

Manifest marks reference intervals verified.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_morning.mp4)

### local_hello_how_you

Prediction: And then I'm going to take the brush and I'm going to take the brush.

Previous CTC replay, phase zero: ['KNOW', 'HOW', 'MAKE']

Existing gloss annotation: ['HELLO', 'HOW', 'YOU']

Manifest does NOT mark intervals verified; no new expert annotation check. Clip-level reference only; no word timings supplied.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/local_hello_how_you.mp4)

### o5s5_lg_hello

Prediction: Hi again, I'm Robert Segundo, and I'm going to talk to you about how to do a flat iron.

Previous CTC replay, phase zero: ['WHY', 'NIGHT']

Existing gloss annotation: None

Partial O5S5 annotation; original ID gloss shown in parentheses. No new expert correction.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/o5s5_lg_hello.mp4)

### o5s5_lg_when

Prediction: I'm going to tell you a little bit about the outer coat.

Previous CTC replay, phase zero: ['WHEN', 'WHEN']

Existing gloss annotation: None

Partial O5S5 annotation; original ID gloss shown in parentheses. No new expert correction.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/o5s5_lg_when.mp4)

## Decision boundary

Keep this as a separate challenger. Do not promote or start training automatically. Review source-grounded meaning and repetition in the displayed outputs; compare with the prior CTC replay. The current 100-gloss pipeline has no matching open-vocabulary English validation benchmark, so these scores are not a head-to-head improvement percentage.

Native online-demo preprocessing was used; this is not a reproduction of the paper’s pre-extracted-pose benchmark. The released model’s pretraining and checkpoint-selection overlap beyond declared splits is not independently audited.

[Official code](https://github.com/ZechengLi19/Uni-Sign), [released weights](https://huggingface.co/ZechengLi19/Uni-Sign), [validation mirror](https://huggingface.co/datasets/aipieces/How2Sign).
