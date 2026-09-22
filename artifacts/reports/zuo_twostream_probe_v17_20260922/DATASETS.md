# Zuo Online / TwoStream data register — 2026-09-22

## Sources for later experiments

| Source | Supervision / role | Access and reuse notes |
| --- | --- | --- |
| [PHOENIX-2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) | German sign-language weather videos, gloss sequences and German translations | Official page advertises a39GB archive;25FPS,210×260 interpreter crops. Preserve official splits and source permissions. Not ASL labels. |
| [PHOENIX-2014](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/) | Related German weather recognition benchmark | Overlaps the PHOENIX family; do not count as independent additional coverage without source reconciliation. |
| [CSL-Daily](http://home.ustc.edu.cn/~zhouh156/dataset/csl-daily/) | Chinese Sign Language continuous recognition/translation | Official TwoStream instructions require an agreement submission. Site timed out in this check. No acquisition or agreement submission performed. |
| Kinetics-400 / WLASL | Generic action / isolated ASL initialization in the TwoStream release | Initializer provenance matters; generic K400 weights alone are not a trained online sign/blank model. Do not equate WLASL labels with our exact100-gloss mapping. |
| COCO-WholeBody | HRNet pose detector supervision | Detector assets, not sign/transition labels. The recognizer expects the selected HRNet body/hand/mouth keypoint layout and confidences. |

The [official online method](https://github.com/FangyunWei/SLRT/tree/main/Online/CSLR) builds its isolated dictionary from continuous training videos using a trained TwoStream teacher. Published metadata groups contextual crops by source sign and includes blank examples. Those interval boundaries are teacher-derived pseudo-labels, not a universal manually annotated transition corpus.

The [TwoStream release](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) provides HRNet-extracted keypoints. Our Apple Vision landmarks and SHuBERT's384-dimensional DINO streams cannot be passed into that input unchanged.

## Recovered author-hosted files

Original university SharePoint links return404. Author comments on [issue106](https://github.com/FangyunWei/SLRT/issues/106#issuecomment-4915936329) and [issue107](https://github.com/FangyunWei/SLRT/issues/107#issuecomment-5452405317) point to [a replacement Google Drive folder](https://drive.google.com/drive/folders/1U-BK7R-fMLmkSDq2M7GpHT9J4aKK9TFa).

- `EMNLP24_Online.zip`:2,705,697,708bytes. Contains PHOENIX and CSL online checkpoints, vocabulary and derived interval metadata.
- `ckpts.zip`:21,312,474,831bytes. Contains assorted older model/features. The PHOENIX-2014T TwoStream S2G checkpoint entry is an error-text placeholder, not a usable checkpoint. Online inference can use its own recovered model without that teacher.
- `data.zip`: listed for future review; not downloaded or admitted.

Archive central directories were read using HTTP byte ranges. Only the PHOENIX online checkpoint(456,022,647bytes) and its vocabulary(12,551bytes) were fetched from the online archive. No train/dev/test interval payloads, feature datasets, or raw signing videos were downloaded. Inventories, URLs and hashes are saved alongside this file.

Later admission still needs a source/split/identity/label contract. These datasets may supply useful initialization or boundary supervision, but foreign gloss heads cannot be treated as our ASL recognizer. Keep comparison before combined training, as requested.
