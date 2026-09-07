#!/usr/bin/env python3
"""Extract metric landmark templates from official ISWA hand photographs."""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from active.v17.avatar_rig_v17 import HAND_BONE_LENGTHS, HAND_EDGES
from active.v17.extract_mediapipe_v17 import MediaPipeHybridDetector, DEFAULT_MODEL_PATH
from active.v17.schema_mediapipe_v17 import MediaPipeV17Config
from scripts.render_rigged_avatar_v17 import sha256


def normalize_template(points):
    """Put a photographed hand in a right-hand local palm frame with metric bones."""
    points = np.asarray(points, np.float32)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError('finite hand landmarks [21,3] required')
    along = points[9] - points[0]
    across = points[5] - points[17]
    along /= max(float(np.linalg.norm(along)), 1e-8)
    across -= np.dot(across, along) * along
    if np.linalg.norm(across) < 1e-7:
        raise ValueError('photographed palm frame is degenerate')
    across /= np.linalg.norm(across)
    normal = np.cross(across, along)
    local = (points - points[0]) @ np.stack((across, along, normal), axis=1)
    output = np.zeros((21, 3), np.float32)
    for index, (parent, child) in enumerate(HAND_EDGES):
        direction = local[child] - local[parent]
        length = float(np.linalg.norm(direction))
        if length < 1e-7:
            raise ValueError('photographed hand has a collapsed bone')
        output[child] = output[parent] + direction / length * HAND_BONE_LENGTHS[index]
    return output


def metric_oriented_template(points, preprocessing_degrees):
    """Undo detector image rotation and preserve the photographed palm orientation."""
    points = np.asarray(points,np.float32).copy()
    if points.shape != (21,3) or not np.isfinite(points).all():
        raise ValueError('finite hand landmarks [21,3] required')
    angle=np.deg2rad(preprocessing_degrees)
    transform=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    points[:,:2]=(np.linalg.inv(transform)@points[:,:2].T).T
    points=(points-points[0])*[1,-1,-1]
    output=np.zeros((21,3),np.float32)
    for index,(parent,child) in enumerate(HAND_EDGES):
        direction=points[child]-points[parent]
        length=float(np.linalg.norm(direction))
        if length < 1e-7:
            raise ValueError('photographed hand has a collapsed bone')
        output[child]=output[parent]+direction/length*HAND_BONE_LENGTHS[index]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path,
        default=Path('artifacts/reports/signwriting_gloss_bank_v17_v5b/report.json'))
    parser.add_argument('--photos', type=Path,
        default=Path('artifacts/tools/iswa_hand_references/first_candidate'))
    parser.add_argument('--extra-views', nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    views = sorted({symbol['key'][:5] for row in inventory['notation_inventory']
        for symbol in row['candidates'][0]['notation']['symbols'] if symbol['category'] == 'hand'} |
        set(args.extra_views))
    required = sorted({view[:4] for view in views})
    detector = MediaPipeHybridDetector(DEFAULT_MODEL_PATH, MediaPipeV17Config(
        include_apple_auxiliary=False, minimum_hand_detection_confidence=.1,
        minimum_hand_presence_confidence=.1, minimum_hand_tracking_confidence=.1))
    candidates = {view: [] for view in views}
    try:
        for path in sorted(args.photos.glob('*.png')):
            view = 'S' + path.stem[:4]
            if view not in candidates:
                continue
            source = cv2.resize(cv2.imread(str(path)),None,fx=8,fy=8,interpolation=cv2.INTER_CUBIC)
            side=max(source.shape[:2])*2
            canvas=np.full((side,side,3),255,np.uint8)
            y,x=(side-source.shape[0])//2,(side-source.shape[1])//2
            canvas[y:y+source.shape[0],x:x+source.shape[1]]=source
            for degrees in range(0,360,45):
                matrix=cv2.getRotationMatrix2D((side/2,side/2),degrees,1)
                image=cv2.warpAffine(canvas,matrix,(side,side),borderValue=(255,255,255))
                detector.reset_sequence()
                hands=detector.detect(image,False,False).hands
                if hands:
                    hand=max(hands,key=lambda value:value.score)
                    try:
                        oriented=metric_oriented_template(hand.world_xyz,degrees)
                    except ValueError:
                        continue
                    candidates[view].append((hand.score,path,hand.chirality,degrees,oriented))
    finally:
        detector.close()
    missing = [view for view, rows in candidates.items() if not rows]
    if missing:
        raise ValueError('no detected official hand photo for: ' + ', '.join(missing))
    chosen = {view:max(rows,key=lambda row:row[0]) for view,rows in candidates.items()}
    bases={base:max((row for view,rows in candidates.items() if view[:4]==base for row in rows),
                    key=lambda row:row[0]) for base in required}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'templates.npz', bases=np.asarray(required),
        templates=np.stack([normalize_template(bases[base][4]) for base in required]),
        views=np.asarray(views),oriented_templates=np.stack([chosen[view][4] for view in views]))
    report = dict(format='slt_iswa_hand_templates_v17', bases=len(required),views=len(views),
        inventory_sha256=sha256(args.inventory), model_sha256=sha256(DEFAULT_MODEL_PATH), rows=[
            dict(view=view,base=view[:4],photo=str(chosen[view][1]),photo_sha256=sha256(chosen[view][1]),
                 detected_chirality=chosen[view][2],confidence=float(chosen[view][0]),
                 preprocessing_degrees=chosen[view][3],successful_rotations=len(candidates[view])) for view in views],
        limitations=['MediaPipe infers depth from one official photograph per handshape',
                     'Templates encode handshape only; FSW fill, rotation and placement are separate',
                     'Official photographs and detector output still require avatar visual review'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(bases=len(required),views=len(views),
        minimum_confidence=min(r['confidence'] for r in report['rows'])),indent=2))


if __name__ == '__main__':
    main()
