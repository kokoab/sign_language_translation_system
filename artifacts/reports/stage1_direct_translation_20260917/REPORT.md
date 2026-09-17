# Stage 1 direct-translation experiment

**Completed:** our Stage 1 encoder -> projection -> mT5, with isolated classification retained. No CTC or repeat suppression.

| System | BLEU | chrF |
|---|---:|---:|
| initialized hybrid | 0.74 | 16.65 |
| trained hybrid | 1.08 | 18.99 |
| zero-visual control | 0.78 | 16.01 |
| unchanged Uni-Sign | 8.21 | 38.06 |

Scores cover the same 12 paired development utterances. Nine short/webcam recordings remain qualitative; their gloss labels are not English reference translations.

## Isolated-sign retention

| Source | Before | After |
|---|---:|---:|
| citizen (378 clips) | 95.24% | 95.24% |
| semlex (978 clips) | 85.28% | 85.07% |

These are existing development subsets, not official Citizen test accuracy.

## Interpretation

The pilot improves both paired text metrics over its initialization and beats the zero-visual chrF control, while isolated accuracy stays within two percentage points. This supports a larger independent evaluation, not deployment.

The zero-visual control preserves sequence lengths and masks but replaces projected motion features with zeros. Similar outputs/scores would weaken evidence that translation uses the signing. Repetition is not an automatic rejection gate.

Training adaptation excludes evaluation signers 1 and 2; inherited mT5 pretraining is not certified signer-disjoint. The checkpoint is initialized from the released Uni-Sign ASL text component, with our own visual encoder. This does not reproduce Uni-Sign’s pose encoder or constitute a fresh large-scale pretraining run.

Evaluation is at utterance completion. No streaming, iPhone, or unseen-domain accuracy claim. No live defaults changed.

## Predictions and references

### -d5dN54tH2E_0-1-rgb_front

Reference: We're going to work on a arm drill that will help you have graceful hand movements in front of you.

Before training: You're going to have your hands on the floor, your hands are going to be on the floor, your feet are going to be on the floor, your feet are going to be on the floor, your feet are going to be on the floor.

After training: You want to make sure that at the very begining and the end of every training session you start and end on a flat spot.

Unchanged Uni-Sign: The best way to do that is to actually move the body pretty much like you're dribbling.

Zero visual features: The next step in recovery is to step on the quick Thai curry.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_0-1-rgb_front.mp4)

### -d5dN54tH2E_1-1-rgb_front

Reference: I call it painting the wall.

Before training: One, two, three, four.

After training: The two fingers are going to be up, they're going to be up.

Unchanged Uni-Sign: It's called a paint tool.

Zero visual features: The best time to do this is April or May of every year.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_1-1-rgb_front.mp4)

### -d5dN54tH2E_10-1-rgb_front

Reference: So we're going to go up and down; let's switch hands, down and up; down and up.

Before training: It's very important that you keep your hands in line.

After training: I am going to teach you how to kneel at the penalty kick.

Unchanged Uni-Sign: So we're going to move one, two, three, four.

Zero visual features: That is the topic of this one, this is going to be politics.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_10-1-rgb_front.mp4)

### -f1_kdl050s_0-1-rgb_front

Reference: In this clip we are going to talk about dangers for these birds in the household and otherwise.

Before training: So you're going to take your left hand, and you're going to take your right hand, and you're going to take your left hand.

After training: In order to get our imagery ready, we're going to need something to put it on.

Unchanged Uni-Sign: In this clip we're going to talk about how to get your birds dangerous within the house and the other things like that.

Zero visual features: The next step in recovery is to step on the quick Thai curry.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_0-1-rgb_front.mp4)

### -f1_kdl050s_1-1-rgb_front

Reference: The number one loss for these birds, is flight.

Before training: So we're going to go ahead and take the hoop and we're going to go ahead and take the hoop.

After training: MIKE LOPEZ: So, next we're going to go on to steaming your milk for your latte.

Unchanged Uni-Sign: The number one negative for birds is they're flying.

Zero visual features: The best time to do this is April or May of every year.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_1-1-rgb_front.mp4)

### -f1_kdl050s_10-1-rgb_front

Reference: You need to be very careful when cleaning their cages if these birds are flighted.

Before training: So you're going to have a very powerful wave.

After training: You just have to take extra care when handling your tool and the chord.

Unchanged Uni-Sign: You should be careful and clean them carefully for the other thing.

Zero visual features: The best time to do this is April or May of every year.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_10-1-rgb_front.mp4)

### 0pKzG0RRUz4_0-2-rgb_front

Reference: Another technique is you can braid the hair on first and then start wrapping cause some people have really short hair and we can create the look with adding the hair end.

Before training: Then you're going to take your hand and you're going to punch it out like this and then you're going to punch it out like this and then you're going to punch it out like

After training: Choosing a saddle can be not only a difficult decision, but one that requires a lot of time and patience in shopping.

Unchanged Uni-Sign: Another technique you can do to round off your hair, the first thing that you can do is start spraying and spraying because some people really youth hair when they can create a crease in the hair.

Zero visual features: A lot of athletes have a pretty big backhand spring, so they are starting their back tuck, and they are starting their back tuck.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_0-2-rgb_front.mp4)

### 0pKzG0RRUz4_3-2-rgb_front

Reference: I really like that.

Before training: It's very easy to do that.

After training: That's why I became a security officer.

Unchanged Uni-Sign: And, I like that from a spoon.

Zero visual features: That is the topic of this one, this is going to be politics.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_3-2-rgb_front.mp4)

### 0pKzG0RRUz4_4-2-rgb_front

Reference: You can get instant length, which I love.

Before training: So you're going to want to make sure that you're getting all of the grooves in there, and you're going to want to make sure that you're getting all of the grooves in there.

After training: Instead of taking it yourself, you want to make sure that you have a good....good anchor point.

Unchanged Uni-Sign: You can get a little bit of a tail that I love.

Zero visual features: The next step in recovery is to step on the quick Thai curry.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_4-2-rgb_front.mp4)

### 0zvsqf23tmw_1-2-rgb_front

Reference: And in this segment, I'm going to just start paint in the background color.

Before training: Forward, back; forward, back; forward, back; forward, back; forward, back; forward, back; forward, back; forward, back; forward, back; forward, back; forward, back; forward, back.

After training: Sometimes, the best thing to do is to avoid the paint smell.

Unchanged Uni-Sign: In this series we're going to start by doing the painting of background color.

Zero visual features: The next step in recovery is to step on the quick Thai curry.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_1-2-rgb_front.mp4)

### 0zvsqf23tmw_10-2-rgb_front

Reference: I'm going to start on the edge so I can get the edge painted in against the skyline.

Before training: He's going to take a few seconds, and he's going to take a few seconds, and he's going to take a few seconds, and he's going to

After training: When it starts to come up on the wax paper you just kind of press it down a little more and it will spread it long.

Unchanged Uni-Sign: We're going to start with the edge of the telescope and we're going to start with the shadow of the ceiling.

Zero visual features: A lot of athletes have a pretty big backhand spring, so they are starting their back tuck, and they are starting their back tuck.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_10-2-rgb_front.mp4)

### 0zvsqf23tmw_11-2-rgb_front

Reference: I'm just doing a dark hill here.

Before training: You're going to hit the ball in the middle.

After training: From here, you're going to bring your right ankle.

Unchanged Uni-Sign: I'm not going to cut anything out of it.

Zero visual features: That is the topic of this one, this is going to be politics.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_11-2-rgb_front.mp4)

### webcam_i

Reference: No verified English reference

Before training: It's very important to keep it in place.

After training: Our next stretch is going to be a side stretch.

Unchanged Uni-Sign: He's going to go to the other side.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_i.mp4)

### webcam_hello

Reference: No verified English reference

Before training: This is a very important tool to use when you're using a wheelchair.

After training: Make sure that the eyebrows are balanced and not falling out.

Unchanged Uni-Sign: I'm going to show you a little bit of what it looks like.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_hello.mp4)

### webcam_eat

Reference: No verified English reference

Before training: Forward, back, forward, back.

After training: Our next clip is going to be a little bit more advanced.

Unchanged Uni-Sign: I'm sorry, I'm sorry.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_eat.mp4)

### asllrp_night_time

Reference: No verified English reference

Before training: It's going to be a little faster.

After training: Glove - you need one glove.

Unchanged Uni-Sign: We're going to start off with this one.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_night_time.mp4)

### asllrp_time_friend

Reference: No verified English reference

Before training: David: Yes.

After training: I'm going to show you how to use a belt.

Unchanged Uni-Sign: Do you have a good time?

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_time_friend.mp4)

### asllrp_morning

Reference: No verified English reference

Before training: So, it's going to be a little bit faster, and it's going to be a little bit faster, and it's going to be a little bit faster, and it's going to be a little

After training: But if you have a white piece of paper that you can use for this, you want to make sure that there is something on there.

Unchanged Uni-Sign: I guess in the bedroom, I guess it's easier to guess the hours.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_morning.mp4)

### local_hello_how_you

Reference: No verified English reference

Before training: Okay, now we're going to go ahead and start turning.

After training: I'm going to use my mascara brush and I'm going to use it in a deep fryer.

Unchanged Uni-Sign: And then I'm going to take the brush and I'm going to take the brush.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/local_hello_how_you.mp4)

### o5s5_lg_hello

Reference: No verified English reference

Before training: You're going to want to make sure that you're getting the right spot.

After training: Hi, I'm going to share with you the tips on how to throw a curve ball.

Unchanged Uni-Sign: Hi again, I'm Robert Segundo, and I'm going to talk to you about how to do a flat iron.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/o5s5_lg_hello.mp4)

### o5s5_lg_when

Reference: No verified English reference

Before training: Inhale, exhale, inhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, exhale, ex

After training: Then you have got a strap at the bottom, you can plug it in.

Unchanged Uni-Sign: I'm going to tell you a little bit about the outer coat.

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/o5s5_lg_when.mp4)


## Result review — 2026-09-18

All 20 epochs completed. Initial/final/zero-visual BLEU and chrF were independently recomputed from saved predictions and match summary.json. Despite improvement over initialization, final BLEU 1.076 / chrF 18.989 remain far below unchanged Uni-Sign 8.206 / 38.059 on the same 12 references. Inspected examples introduce unrelated content: the reference about graceful hand movements is translated into starting and ending training on a flat spot; the hand-switching reference is translated into kneeling at a penalty kick. This is not successful translation. The automated descriptive checks are too weak to establish useful visual grounding or generalization. Isolated development accuracy was retained (Citizen unchanged; SemLex two fewer correct).

The English comparison stopped before trim evaluation or BART training at the startup storage gate. Current internal free space is approximately 7.7 GiB, below the 8 GiB requirement; SSD is not mounted. Nothing is currently training. No additional quality conclusion about BART is possible.
