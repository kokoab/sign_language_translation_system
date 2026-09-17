# English model comparison results

Screening only; 12 correlated sentences cannot establish accuracy noninferiority. No production promotion.

| Model/control | BLEU | chrF |
| --- | ---: | ---: |
| full mT5 | 1.076 | 18.989 |
| trimmed mT5 | 1.076 | 18.989 |
| initialized BART | 0.332 | 9.052 |
| trained BART | 1.061 | 17.464 |
| BART zero visual | 0.390 | 12.388 |
| BART mismatched visual | 1.287 | 15.652 |
| full mT5 mismatched visual | 1.522 | 16.670 |

## Isolated development retention

| Subset | Before | Full mT5 | BART |
| --- | ---: | ---: | ---: |
| citizen | 95.24% | 95.24% | 94.71% |
| semlex | 85.28% | 85.07% | 85.99% |

## Descriptive checks

{
  "chrf_within_1": false,
  "bleu_within_half": true,
  "isolated_within_1pp": true,
  "beats_zero_visual": true,
  "beats_mismatched_visual": true
}

No automatic deployment. Expanded independent signer/source evaluation and semantic review remain required. Inherited mT5 pretraining is not certified signer-disjoint. No mobile timing or accuracy claim. Intended repetitions remain allowed.

BART used the same original Stage-1 checkpoint and training rows, full coverage, joint isolated CE, 20 fixed epochs. BART adds English text pretraining; it does not inherit the ASL mT5 weights. The trimmed checkpoint was evaluated without retraining. Timing/preflight and hashes are saved alongside this report.

## Video predictions

### -d5dN54tH2E_0-1-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_0-1-rgb_front.mp4)

Reference: We're going to work on a arm drill that will help you have graceful hand movements in front of you.

Full mT5: You want to make sure that at the very begining and the end of every training session you start and end on a flat spot.

Trimmed mT5: You want to make sure that at the very begining and the end of every training session you start and end on a flat spot.

BART: I'm soaking my feet in a plain plastic basin with the Epsom salt in bath temperature water.

### -d5dN54tH2E_1-1-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_1-1-rgb_front.mp4)

Reference: I call it painting the wall.

Full mT5: The two fingers are going to be up, they're going to be up.

Trimmed mT5: The two fingers are going to be up, they're going to be up.

BART: As far as tools go, that's your basics.

### -d5dN54tH2E_10-1-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-d5dN54tH2E_10-1-rgb_front.mp4)

Reference: So we're going to go up and down; let's switch hands, down and up; down and up.

Full mT5: I am going to teach you how to kneel at the penalty kick.

Trimmed mT5: I am going to teach you how to kneel at the penalty kick.

BART: Press up on the left side and all the way up.

### -f1_kdl050s_0-1-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_0-1-rgb_front.mp4)

Reference: In this clip we are going to talk about dangers for these birds in the household and otherwise.

Full mT5: In order to get our imagery ready, we're going to need something to put it on.

Trimmed mT5: In order to get our imagery ready, we're going to need something to put it on.

BART: In this segment we're going to cover the overhead serve and this is very similar to a tennis serve or an empty goal.

### -f1_kdl050s_1-1-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_1-1-rgb_front.mp4)

Reference: The number one loss for these birds, is flight.

Full mT5: MIKE LOPEZ: So, next we're going to go on to steaming your milk for your latte.

Trimmed mT5: MIKE LOPEZ: So, next we're going to go on to steaming your milk for your latte.

BART: The king is thirteen by himself.

### -f1_kdl050s_10-1-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/-f1_kdl050s_10-1-rgb_front.mp4)

Reference: You need to be very careful when cleaning their cages if these birds are flighted.

Full mT5: You just have to take extra care when handling your tool and the chord.

Trimmed mT5: You just have to take extra care when handling your tool and the chord.

BART: First off I'm going to start off with a little baby powder.

### 0pKzG0RRUz4_0-2-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_0-2-rgb_front.mp4)

Reference: Another technique is you can braid the hair on first and then start wrapping cause some people have really short hair and we can create the look with adding the hair end.

Full mT5: Choosing a saddle can be not only a difficult decision, but one that requires a lot of time and patience in shopping.

Trimmed mT5: Choosing a saddle can be not only a difficult decision, but one that requires a lot of time and patience in shopping.

BART: I arrived early in a rural part of New Mexico, to do cancer screening, and opened up the doors, and lo and behold, there were two clients already waiting there for me.

### 0pKzG0RRUz4_3-2-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_3-2-rgb_front.mp4)

Reference: I really like that.

Full mT5: That's why I became a security officer.

Trimmed mT5: That's why I became a security officer.

BART: And then you have got a good staple right there nice and secure.

### 0pKzG0RRUz4_4-2-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0pKzG0RRUz4_4-2-rgb_front.mp4)

Reference: You can get instant length, which I love.

Full mT5: Instead of taking it yourself, you want to make sure that you have a good....good anchor point.

Trimmed mT5: Instead of taking it yourself, you want to make sure that you have a good....good anchor point.

BART: So again, laying out some of my mixed medium onto my plexiglass board.

### 0zvsqf23tmw_1-2-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_1-2-rgb_front.mp4)

Reference: And in this segment, I'm going to just start paint in the background color.

Full mT5: Sometimes, the best thing to do is to avoid the paint smell.

Trimmed mT5: Sometimes, the best thing to do is to avoid the paint smell.

BART: So, as far as color changing goes, she also--many chameleons go through a breeding stage.

### 0zvsqf23tmw_10-2-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_10-2-rgb_front.mp4)

Reference: I'm going to start on the edge so I can get the edge painted in against the skyline.

Full mT5: When it starts to come up on the wax paper you just kind of press it down a little more and it will spread it long.

Trimmed mT5: When it starts to come up on the wax paper you just kind of press it down a little more and it will spread it long.

BART: So again, laying out some of my mixed medium onto my plexiglass board.

### 0zvsqf23tmw_11-2-rgb_front

[Video](/Users/frnzlo/Documents/machine_learning/SLT/data/local/unisign_asl_baseline_20260916/clips/0zvsqf23tmw_11-2-rgb_front.mp4)

Reference: I'm just doing a dark hill here.

Full mT5: From here, you're going to bring your right ankle.

Trimmed mT5: From here, you're going to bring your right ankle.

BART: Staging should be an interval part of your marketing strategy.

### webcam_i

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_i.mp4)

Reference: No verified English reference

Full mT5: Our next stretch is going to be a side stretch.

Trimmed mT5: Our next stretch is going to be a side stretch.

BART: The king is thirteen by himself.

### webcam_hello

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_hello.mp4)

Reference: No verified English reference

Full mT5: Make sure that the eyebrows are balanced and not falling out.

Trimmed mT5: Make sure that the eyebrows are balanced and not falling out.

BART: Move the seasoning around.

### webcam_eat

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/webcam_eat.mp4)

Reference: No verified English reference

Full mT5: Our next clip is going to be a little bit more advanced.

Trimmed mT5: Our next clip is going to be a little bit more advanced.

BART: Put it on the fingertip, move it over, and pinch it off.

### asllrp_night_time

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_night_time.mp4)

Reference: No verified English reference

Full mT5: Glove - you need one glove.

Trimmed mT5: Glove - you need one glove.

BART: We're going to utilize straps, blocks, different variations of your seated poses.

### asllrp_time_friend

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_time_friend.mp4)

Reference: No verified English reference

Full mT5: I'm going to show you how to use a belt.

Trimmed mT5: I'm going to show you how to use a belt.

BART: It's different, it's a whole different world.

### asllrp_morning

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/asllrp_morning.mp4)

Reference: No verified English reference

Full mT5: But if you have a white piece of paper that you can use for this, you want to make sure that there is something on there.

Trimmed mT5: But if you have a white piece of paper that you can use for this, you want to make sure that there is something on there.

BART: I'm holding it in the fingers, basic position, and depending on what I do, it changes its position in my hand, and that's key.

### local_hello_how_you

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/local_hello_how_you.mp4)

Reference: No verified English reference

Full mT5: I'm going to use my mascara brush and I'm going to use it in a deep fryer.

Trimmed mT5: I'm going to use my mascara brush and I'm going to use it in a deep fryer.

BART: So now we have sliced our mirliton in half again chayote in some parts of the country.

### o5s5_lg_hello

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/o5s5_lg_hello.mp4)

Reference: No verified English reference

Full mT5: Hi, I'm going to share with you the tips on how to throw a curve ball.

Trimmed mT5: Hi, I'm going to share with you the tips on how to throw a curve ball.

BART: So, you can see, I'm getting and I'm hitting the ball and you end up like this.

### o5s5_lg_when

[Video](/Users/frnzlo/Documents/machine_learning/SLT/artifacts/reports/stage2_research_review_20260915/media/o5s5_lg_when.mp4)

Reference: No verified English reference

Full mT5: Then you have got a strap at the bottom, you can plug it in.

Trimmed mT5: Then you have got a strap at the bottom, you can plug it in.

BART: We're moving on to the next step in our quick Thai curry.


## Independent result review — 2026-09-20

BART completed all20 epochs, training time10552.19 seconds (2h56m). Final BART, trimmed-mT5 and mismatched-visual metrics independently recomputed from predictions and match summary.json. Vocabulary trimming reduced total parameters589.39M→303.52M (48.5%) and preserved exact predictions on all21 checked clips, not just aggregate scores. This preserves the full hybrid’s poor translations; it is not evidence of useful translation or general equivalence.

BART uses146.41M parameters (75.2% fewer) but chrF17.464 is1.524 below full-mT5 18.989, outside the proposed1-point retention margin. Its BLEU1.061 is also much lower than unchanged Uni-Sign8.206. Incorrectly paired visual inputs yield BLEU1.287, higher than correctly paired inputs, despite lower chrF15.652. Thus the automated chrF-only grounding check does not establish reliable interpretation. Inspected output describes soaking feet when the reference discusses graceful arm/hand movements. No hybrid should be promoted.

Isolated development retention: Citizen358/378 (94.71%) versus360/378 initially; SemLex841/978 (85.99%) versus834/978 initially. This preserves isolated performance reasonably but does not resolve connected-sign translation. These are development subsets, not official-test accuracy. The next useful work is diagnosis of visual-text alignment/generalization and supervision, rather than claiming model-size reduction solved the task. No further training launched.
