# Native review: generated Stage-2 phrase pilot

These are **synthetic landmark-avatar review videos**, not genuine recordings or
validated training truth. Each MP4 shows the same requested phrase in three generated
signing voices: Aster, Cobalt, and Juniper.

Open [the video gallery](index.html) or [the contact sheet](contact_sheet.png), then
review each voice separately. Record decisions in [review.csv](review.csv). An item
must not enter training unless a native signer marks the lexical sequence, every
transition, and the overall sample acceptable. Nothing in this directory is eligible
for validation or testing.

| ID | Requested glosses | Video |
| --- | --- | --- |
| 01 | `HELLO YOU READY` | [Play MP4](generated_01/generated_01.mp4) |
| 02 | `GOOD MORNING FRIEND` | [Play MP4](generated_02/generated_02.mp4) |
| 03 | `GOOD NIGHT FAMILY` | [Play MP4](generated_03/generated_03.mp4) |
| 04 | `PLEASE WAIT` | [Play MP4](generated_04/generated_04.mp4) |
| 05 | `PLEASE STOP` | [Play MP4](generated_05/generated_05.mp4) |
| 06 | `I HELP YOU` | [Play MP4](generated_06/generated_06.mp4) |
| 07 | `YOU HELP I` | [Play MP4](generated_07/generated_07.mp4) |
| 08 | `I NEED WATER` | [Play MP4](generated_08/generated_08.mp4) |
| 09 | `I NEED DOCTOR NOW` | [Play MP4](generated_09/generated_09.mp4) |
| 10 | `YOU WANT EAT` | [Play MP4](generated_10/generated_10.mp4) |

## What to inspect

For each of the three voices, verify:

1. Every requested sign is the intended pinned lexical variant.
2. The handshape and movement remain readable through each yellow transition interval.
3. Hands do not pop, disappear, teleport, freeze unnaturally, or change handedness.
4. Rhythm, body posture, gaze, and coarticulation look plausible as one utterance.
5. The complete sequence is acceptable as training material.

The audited [manifest](manifest.json) contains source-voice mixtures, transition spans,
raw-landmark hashes, video hashes, and machine diagnostics. All 30 voice/phrase renders
passed content-retention and structural-continuity gates. These gates do not establish
linguistic or perceptual naturalness; that decision belongs to native review.

