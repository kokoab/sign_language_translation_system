"""Metric avatar retargeting for v17 review artifacts.

The v17 third spatial channel is a log-scale proxy, not camera depth.  This module
therefore uses it only as a small, bounded ordering cue.  Avatar anatomy lives in
metric model space and never inherits detector hand scale.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
import re
from xml.etree import ElementTree as ET

import numpy as np


HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)

# Adult-proportioned review rig in metres. Values define the avatar, not a claim
# about the source signer's anatomy.
HAND_BONE_LENGTHS = np.asarray((
    .041, .032, .024, .020, .087, .045, .027, .021, .091, .050,
    .031, .023, .086, .046, .028, .021, .076, .038, .023, .018,
), dtype=np.float32)
UPPER_ARM_LENGTH = .295
FOREARM_LENGTH = .265


def parse_signwriting_signbox(fsw):
    """Preserve one FSW signbox's spatial notation; parsing is not motion support."""
    key = r'S[123][0-9a-f]{2}[0-5][0-9a-f]'
    match = re.fullmatch(r'(?:A(?P<sort>(?:' + key + r')+))?'
        r'(?P<lane>[BLMR])(?P<x>[0-9]{3})x(?P<y>[0-9]{3})'
        r'(?P<spatial>(?:' + key + r'[0-9]{3}x[0-9]{3})+)', fsw) if isinstance(fsw, str) else None
    if match is None:
        raise ValueError('expected one FSW signbox')
    sort = re.findall(key, match['sort'] or '')
    spatial = re.findall('(' + key + r')([0-9]{3})x([0-9]{3})', match['spatial'])
    if any(not 0x100 <= int(s[1:4], 16) <= 0x38b for s in sort + [s[0] for s in spatial]):
        raise ValueError('symbol base outside ISWA 2010')
    symbols = []
    for symbol, x, y in spatial:
        base = int(symbol[1:4], 16)
        category = next(name for stop, name in ((0x204, 'hand'), (0x215, 'contact'),
            (0x229, 'finger_movement'), (0x2f6, 'movement'), (0x2fe, 'dynamics'),
            (0x36c, 'head_face'), (0x37e, 'body'), (0x386, 'location'), (0x38b, 'punctuation'))
            if base <= stop)
        symbols.append(dict(key=symbol, base=symbol[:4], fill=int(symbol[4]),
            rotation=int(symbol[5], 16), x=int(x), y=int(y), category=category))
    return dict(lane=match['lane'], extent=[int(match['x']), int(match['y'])],
                sort_symbols=sort, symbols=symbols)


_PILOT_SYMBOL_PAIRS = {('S10040','S26500'), ('S10620','S22e04'),
                       ('S10620','S23004'), ('S20320','S23004'),
                       ('S15a20','S26500'), ('S15a28','S26500'), ('S15a11','S26500'),
                       ('S14c20','S27206')}


def signwriting_pilot_symbols(fsw, *, repeated):
    """Check the executable spatial pair and frozen lexical repetition."""
    symbols = parse_signwriting_signbox(fsw)['symbols']
    hands = [s['key'] for s in symbols if s['category'] == 'hand']
    movements = [s['key'] for s in symbols if s['category'] == 'movement']
    if len(hands)!=1 or len(movements)!=1:
        raise ValueError('pilot needs exactly one hand and one movement')
    pair=(hands[0],movements[0])
    if pair not in _PILOT_SYMBOL_PAIRS:
        raise ValueError('unsupported pilot symbol pair')
    extras = {s['key'] for s in symbols if s['category'] not in ('hand','movement')}
    allowed = {'S30007','S20500'} if pair == ('S15a11','S26500') else set()
    if extras != allowed:
        raise ValueError('unsupported or missing companion symbols')
    if (movements[0] in ('S23004', 'S27206')) != bool(repeated):
        raise ValueError('dictionary repetition disagrees with locked ASL variant')
    return pair


def signwriting_pilot_sigml(entries):
    """Convert the YOU/NEED dictionary pilot to SiGML for CWASA's Anna avatar.

    This is an unreviewed two-sign bridge, not a general SignWriting translator.
    Placement, thumb posture and timing remain animation assumptions.
    """
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ValueError('a non-empty sequence of pilot entries is required')
    root = ET.Element('sigml')
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('fsw'), str):
            raise ValueError('each pilot entry needs gloss and FSW strings')
        gloss = entry.get('gloss')
        if gloss not in ('YOU', 'NEED'):
            raise ValueError('SiGML pilot supports only YOU and NEED')
        hand, _ = signwriting_pilot_symbols(entry['fsw'], repeated=gloss == 'NEED')
        if hand != ('S10040' if gloss == 'YOU' else 'S10620'):
            raise ValueError('SignWriting handshape disagrees with pilot gloss')
        manual = ET.SubElement(ET.SubElement(root, 'hamgestural_sign', gloss=gloss), 'sign_manual')
        if gloss == 'YOU':
            ET.SubElement(manual, 'handconfig', handshape='finger2', thumbpos='across',
                          extfidir='o', palmor='l')
            ET.SubElement(manual, 'location_bodyarm', location='chest', contact='neutral')
            ET.SubElement(manual, 'directedmotion', direction='o', size='small')
        else:
            # ponytail: Anna calibration gives PIP ~90deg/DIP 0; remeasure for other avatars.
            ET.SubElement(manual, 'handconfig', handshape='finger2', thumbpos='across',
                          bend2='0 3.34 0 0')
            ET.SubElement(manual, 'handconfig', extfidir='u')
            ET.SubElement(manual, 'handconfig', palmor='d')
            ET.SubElement(manual, 'location_bodyarm', location='shoulders',
                          side='right_at', contact='neutral')
            repeat = ET.SubElement(manual, 'rpt_motion', repetition='fromstart')
            target = ET.SubElement(repeat, 'tgt_motion')
            ET.SubElement(target, 'changeposture')
            ET.SubElement(target, 'handconfig', extfidir='o')
            ET.SubElement(target, 'handconfig', palmor='d')
    return ET.tostring(root, encoding='unicode')


def animate_signwriting_pilot(hand, hand_symbol, movement_symbol, *, frames=32,
                              travel_metres=.06, wrist_flex_degrees=65,
                              wrist_position=(-.18,1.10,.28)):
    """Execute the supported dictionary symbol pairs with explicit motion assumptions.

    Right-handed pilot only: S10040/S26500 (forward point, small forward move),
    S10620/S22e04 or S23004 (bent index, palm away, single or double wrist flex).
    S20320/S23004 adds a fist with double wrist flex. A static reference supplies
    hand proportions (and the accepted index presets' thumb), never its trajectory.
    S15a/S265 is a flat-hand push; S14c/S272 is an alternating wrist wave.
    The S15a28 left-hand dictionary pose is normalized to the right signing hand.
    """
    if (hand_symbol, movement_symbol) not in _PILOT_SYMBOL_PAIRS:
        raise ValueError('unsupported pilot hand/orientation/movement combination')
    if not isinstance(frames, int) or frames < 2 or not np.isfinite([travel_metres,wrist_flex_degrees]).all():
        raise ValueError('finite motion parameters and at least two frames required')
    if not 0 <= travel_metres <= .2 or not 0 <= wrist_flex_degrees <= 90:
        raise ValueError('motion outside pilot range')
    hand = np.asarray(hand, np.float32)
    if hand.shape != (21,3) or not np.isfinite(hand).all():
        raise ValueError('finite static reference hand [21,3] required')
    shape = constrain_signwriting_handshape(hand[None], hand_symbol, side=1)[0]
    normal = np.cross(shape[5]-shape[17], shape[9]-shape[0])
    normal /= np.linalg.norm(normal)
    axis = shape[9]-shape[0] if hand_symbol[:4] in ('S203', 'S15a', 'S14c') else shape[6]-shape[5]
    axis /= np.linalg.norm(axis)
    before = np.stack((axis, normal, np.cross(axis,normal)),axis=1)
    target_axis = np.array([0.,0.,1.]) if hand_symbol=='S10040' else np.array([0.,1.,0.])
    target_normal = np.array([1.,0.,0.]) if hand_symbol=='S10040' else np.array([0.,0.,1.])
    after = np.stack((target_axis,target_normal,np.cross(target_axis,target_normal)),axis=1)
    shape = (shape-shape[0]) @ (after @ before.T).T
    wrist = np.asarray(wrist_position,np.float32)
    if wrist.shape!=(3,) or not np.isfinite(wrist).all():
        raise ValueError('finite wrist position [3] required')
    poses=[]
    # ponytail: minimum-jerk strokes; duration, amplitude and lack of holds are
    # animation assumptions to validate against fluent review, not encoded timings.
    for t in np.linspace(0,1,frames):
        if movement_symbol in ('S23004', 'S27206'):
            # Flex, return, flex. A repeated symbol must survive motion synthesis.
            t = 3*t if t < 1/3 else (2-3*t if t < 2/3 else 3*t-2)
        progress=t**3*(10-15*t+6*t*t)
        if movement_symbol=='S26500':
            posed=shape+wrist+[0,0,travel_metres*progress]
        elif movement_symbol == 'S27206':
            # Three alternating lateral wrist strokes; amplitude is a calibration.
            angle = np.deg2rad(wrist_flex_degrees) * (2*progress-1)
            c,s = np.cos(angle),np.sin(angle)
            posed = shape @ np.array([[c,-s,0],[s,c,0],[0,0,1]]).T + wrist
        else:
            angle=np.deg2rad(wrist_flex_degrees)*progress
            c,s=np.cos(angle),np.sin(angle)
            rotation=np.array([[1,0,0],[0,c,-s],[0,s,c]])
            posed=shape @ rotation.T+wrist
        poses.append(posed)
    return np.asarray(poses,np.float32)


def constrain_signwriting_handshape(hands, symbol, *, side):
    """Review presets for index, bent index, fist, flat and spread hands.

    Only handshape is decoded here. Callers must separately execute orientation,
    placement and motion; this function alone is not complete FSW animation.
    """
    if not re.fullmatch(r"S(?:100|106|203|15a|14c)[0-5][0-9a-f]", symbol) or side not in (0, 1):
        raise ValueError("unsupported ISWA handshape or anatomical side")
    output = np.asarray(hands, np.float32).copy()
    if output.ndim != 3 or output.shape[1:] != (21, 3) or not np.isfinite(output).all():
        raise ValueError("finite hands [frames,21,3] required")
    across = output[:, 5] - output[:, 17]
    along = output[:, 9] - output[:, 0]
    normal = np.cross(across, along) * (-1 if side == 0 else 1)
    if (np.linalg.norm(normal, axis=-1) < 1e-7).any():
        raise ValueError("palm frame is degenerate")
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    longitudinal = along / np.linalg.norm(along, axis=-1, keepdims=True)
    transverse = np.cross(longitudinal, normal) * (-1 if side == 0 else 1)
    base = symbol[1:4]
    for first in (5, 9, 13, 17):
        lengths = np.linalg.norm(np.diff(output[:, first:first + 4], axis=1), axis=-1)
        direction = output[:, first] - output[:, 0]
        if base == '14c':
            spread = np.deg2rad({5:12, 9:0, 13:-12, 17:-25}[first])
            direction = longitudinal * np.cos(spread) + transverse * np.sin(spread)
        elif base in ('203', '15a'):
            direction = along.copy()
        elif first == 5:
            direction = output[:, first + 1] - output[:, first]
        direction -= (direction * normal).sum(-1, keepdims=True) * normal
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        if (norm < 1e-7).any():
            raise ValueError("finger direction cannot define a palm-plane axis")
        direction /= norm
        # ponytail: fixed flexion presets; exact joint angles require fluent review.
        if base in ('15a', '14c') or (first == 5 and base == '100'):
            angles = (0, 0, 0)
        elif base == '203':
            angles = (80, 175, 230)
        else:
            angles = (0, 90, 90) if first == 5 else (65, 155, 200)
        for joint, angle in enumerate(np.deg2rad(angles), 1):
            vector = direction * np.cos(angle) + normal * np.sin(angle)
            output[:, first + joint] = output[:, first + joint - 1] + vector * lengths[:, joint - 1, None]
    if base in ('203', '15a', '14c'):
        # S203 reference photos: thumb across the closed fingers, never tucked inside.
        # https://www.signbank.org/iswa/203/203_bs.html
        lengths = np.linalg.norm(np.diff(output[:, 1:5], axis=1), axis=-1)
        thumb = {'203': ((.05,.80,.60),(-.85,.25,.46),(-.94,-.25,.23)),
                 '15a': ((.50,.85,0),(.25,.97,0),(.10,.995,0)),
                 '14c': ((.90,.435,0),(.80,.60,0),(.75,.66,0))}[base]
        for joint, (x, y, z) in enumerate(thumb, 2):
            direction = x * transverse + y * longitudinal + z * normal
            direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
            output[:, joint] = output[:, joint-1] + direction * lengths[:, joint-2, None]
    return output


def constrain_annotated_handshape(hands, selected_fingers, flexion):
    """Apply an explicit lexicon extension constraint to generated hand poses.

    This is lexical animation guidance, not recovered video geometry. Curved/bent
    entries remain untouched; their articulation cannot be inferred from an open
    hand rule. The chosen 3D pointing axis still comes from the observed estimate.
    """
    output = np.asarray(hands, np.float32).copy()
    if flexion != "FullyOpen":
        return output
    for finger, first in (("i", 5), ("m", 9), ("r", 13), ("p", 17)):
        if finger not in selected_fingers:
            continue
        chain = output[:, first:first + 4]
        lengths = np.linalg.norm(np.diff(chain, axis=1), axis=-1)
        direction = chain[:, -1] - chain[:, 0]
        fallback = chain[:, 1] - chain[:, 0]
        direction = np.where((np.linalg.norm(direction, axis=-1) < 1e-5)[:, None], fallback, direction)
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
        direction = _smooth(direction)
        direction /= np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-8)
        for index in range(1, 4):
            chain[:, index] = chain[:, index - 1] + direction * lengths[:, index - 1, None]
    return output


def interpolate_world_hand(left, right, alpha):
    """Rotate the palm together and interpolate finger flexion in its local frame."""
    if not 0 <= alpha <= 1:
        raise ValueError("hand blend fraction outside [0,1]")
    if alpha == 0 or np.array_equal(left, right):
        return left.copy()
    if alpha == 1:
        return right.copy()
    from scipy.spatial.transform import Rotation, Slerp
    bases = []
    for pose in (left, right):
        across = pose[5] - pose[17]
        normal = np.cross(across, pose[9] - pose[0])
        if min(np.linalg.norm(across), np.linalg.norm(normal)) < 1e-8:
            # Degenerate detector estimates have no palm frame; retain finite bone arcs.
            return _interpolate_hand_bones(left, right, alpha)
        across /= np.linalg.norm(across)
        normal /= np.linalg.norm(normal)
        bases.append(np.stack((across, np.cross(normal, across), normal), axis=1))
    rotation = Slerp([0, 1], Rotation.from_matrix(np.stack(bases)))(alpha).as_matrix()
    local = _interpolate_hand_bones((left-left[0]) @ bases[0], (right-right[0]) @ bases[1], alpha)
    return (local @ rotation.T + left[0]*(1-alpha) + right[0]*alpha).astype(np.float32)


def _interpolate_hand_bones(left, right, alpha):
    """Interpolate bone directions on arcs, preserving interpolated edge lengths."""
    result = np.zeros((21, 3), np.float32)
    result[0] = left[0] * (1 - alpha) + right[0] * alpha
    for parent, child in HAND_EDGES:
        first, last = left[child] - left[parent], right[child] - right[parent]
        l1, l2 = np.linalg.norm(first), np.linalg.norm(last)
        if min(l1, l2) < 1e-8:
            direction = first * (1 - alpha) + last * alpha
        else:
            u, v = first / l1, last / l2
            cosine = np.clip(u @ v, -1., 1.)
            angle = np.arccos(cosine)
            if cosine > .9999:
                direction = u * (1 - alpha) + v * alpha
            else:
                tangent = v - cosine * u
                if np.linalg.norm(tangent) < 1e-5:
                    tangent = np.cross(u, np.eye(3)[np.argmin(np.abs(u))])
                tangent /= np.linalg.norm(tangent)
                direction = u * np.cos(alpha * angle) + tangent * np.sin(alpha * angle)
        direction /= max(np.linalg.norm(direction), 1e-8)
        result[child] = result[parent] + direction * ((1 - alpha) * l1 + alpha * l2)
    return result


@dataclass
class RetargetedAvatar:
    shoulders: np.ndarray       # [T,2,3]
    elbows: np.ndarray          # [T,2,3]
    hands: np.ndarray           # [T,2,21,3]
    hand_states: np.ndarray     # [T,2], U12 strings
    source_observed: np.ndarray # [T,2]


def mirror_avatar(rig: RetargetedAvatar) -> RetargetedAvatar:
    """Reflect avatar coordinates and exchange hand/arm identities together."""
    return RetargetedAvatar(*(getattr(rig, name)[:, ::-1] * np.array([-1, 1, 1], np.float32)
                              for name in ('shoulders', 'elbows', 'hands')),
                            rig.hand_states[:, ::-1].copy(), rig.source_observed[:, ::-1].copy())


def prepend_isolated_approach(rig: RetargetedAvatar, frames: int) -> RetargetedAvatar:
    """Add an inferred neutral approach to an isolated clip; leave its core exact.

    Do not apply between connected signs: those start from the preceding pose.
    """
    if not isinstance(frames, int) or frames < 0:
        raise ValueError('approach frames must be a nonnegative integer')
    if frames == 0:
        return rig
    shoulders = np.repeat(rig.shoulders[:1], frames, axis=0)
    elbows = np.repeat(rig.elbows[:1], frames, axis=0)
    hands = np.repeat(rig.hands[:1], frames, axis=0)
    states = np.repeat(rig.hand_states[:1], frames, axis=0).astype('<U16')
    for side in range(2):
        if str(rig.hand_states[0, side]).startswith('rest'):
            continue
        rest = _fallback_rest_hand(side)
        _, wrist = _solve_elbow(shoulders[0, side], rest[0], side)
        rest += wrist - rest[0]
        target = rig.hands[0, side]
        # ponytail: neutral approach is inferred; use measured full-clip motion when available.
        for i in range(frames):
            t = i / frames
            alpha = t * t * (3 - 2 * t)
            hands[i, side] = interpolate_world_hand(rest, target, alpha)
            elbows[i, side], wrist = _solve_elbow(shoulders[i, side], hands[i, side, 0], side)
            hands[i, side] += wrist - hands[i, side, 0]
        states[:, side] = 'approach-assumed'
    return RetargetedAvatar(
        np.concatenate((shoulders, rig.shoulders)), np.concatenate((elbows, rig.elbows)),
        np.concatenate((hands, rig.hands)), np.concatenate((states, rig.hand_states)),
        np.concatenate((np.zeros((frames, 2), dtype=bool), rig.source_observed)))


def append_isolated_release(rig: RetargetedAvatar, frames: int) -> RetargetedAvatar:
    """Reuse the neutral approach backwards for one isolated end-of-utterance release."""
    reversed_rig = RetargetedAvatar(*(getattr(rig, f.name)[::-1] for f in fields(rig)))
    padded = prepend_isolated_approach(reversed_rig, frames)
    result = RetargetedAvatar(*(getattr(padded, f.name)[::-1] for f in fields(padded)))
    if frames:
        states = result.hand_states[-frames:]
        states[states == 'approach-assumed'] = 'release-assumed'
    return result


def connect_avatar_clips(clips: list[RetargetedAvatar], transition_frames: int) -> RetargetedAvatar:
    """Join core motions directly, with no intermediate neutral pose or lock-in hold."""
    if not clips or any(not len(clip.hands) for clip in clips):
        raise ValueError('non-empty avatar clips required')
    if not isinstance(transition_frames, int) or transition_frames < 1:
        raise ValueError('positive transition frame count required')
    parts = [clips[0]]
    for left, right in zip(clips, clips[1:]):
        n = transition_frames
        shoulders = np.empty((n, 2, 3), np.float32)
        elbows = np.empty_like(shoulders)
        hands = np.empty((n, 2, 21, 3), np.float32)
        states = np.full((n, 2), 'transition', dtype='<U16')
        for i in range(n):
            t = (i + 1) / (n + 1)
            alpha = t**3 * (10 - 15*t + 6*t*t)
            shoulders[i] = left.shoulders[-1]*(1-alpha) + right.shoulders[0]*alpha
            for side in range(2):
                hands[i, side] = interpolate_world_hand(left.hands[-1, side], right.hands[0, side], alpha)
                elbows[i, side], wrist = _solve_elbow(shoulders[i, side], hands[i, side, 0], side)
                hands[i, side] += wrist - hands[i, side, 0]
                if all(str(state).startswith('rest') for state in
                       (left.hand_states[-1, side], right.hand_states[0, side])):
                    states[i, side] = 'rest-uncertain'
        parts.extend((RetargetedAvatar(shoulders, elbows, hands, states, np.zeros((n, 2), bool)), right))
    return RetargetedAvatar(*(np.concatenate([getattr(part, f.name) for part in parts])
                              for f in fields(parts[0])))


@dataclass
class RestReference:
    """Same-signer low-motion hand observations, grouped into real pose modes."""
    poses: tuple[np.ndarray, np.ndarray]  # each [modes,21,3] source coordinates
    candidate_counts: tuple[int, int]
    source_paths: tuple[str, ...]


def _smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or len(values) < 2:
        return values.copy()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    weights = np.asarray([1, 2, 3, 2, 1], dtype=np.float32)
    weights = weights[:2 * radius + 1]
    weights /= weights.sum()
    return np.stack([
        (padded[index:index + len(weights)] * weights[:, None]).sum(axis=0)
        for index in range(len(values))
    ])


def hand_states_from_metadata(
    frames: int, metadata: dict, observation_presence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate observed, detector-imputed active, and intentional rest states."""
    observed = np.stack((
        observation_presence[:, :21].sum(axis=1) >= 5,
        observation_presence[:, 21:42].sum(axis=1) >= 5,
    ), axis=1)
    participating = np.zeros((frames, 2), dtype=bool)
    timeline = metadata.get("timeline", [])
    gloss_rows = [row for row in timeline if row.get("kind") == "gloss"]
    for row in gloss_rows:
        participating[int(row["start"]):int(row["stop"])] = row["hand_participation"]
    for row in timeline:
        if row.get("kind") != "transition":
            continue
        start, stop = int(row["start"]), int(row["stop"])
        before = next((g for g in reversed(gloss_rows) if int(g["stop"]) <= start), None)
        after = next((g for g in gloss_rows if int(g["start"]) >= stop), None)
        left = np.asarray(before["hand_participation"] if before else (False, False))
        right = np.asarray(after["hand_participation"] if after else (False, False))
        participating[start:stop] = left | right
    # Metadata predates explicit state labels; observed evidence still wins.
    participating |= observed
    states = np.full((frames, 2), "rest", dtype="<U14")
    states[participating] = "imputed-active"
    states[participating & observed] = "observed"
    return states, observed


def _source_to_world(xyz: np.ndarray) -> np.ndarray:
    output = np.empty_like(xyz, dtype=np.float32)
    # v17 XY is already isotropic and measured in shoulder widths. Map both axes
    # with this avatar's shoulder width instead of stretching them independently.
    output[..., 0] = xyz[..., 0] * .335425
    output[..., 1] = 1.285 - xyz[..., 1] * .335425
    # Review-only signing plane in front of the mesh (torso front is < .16 m).
    # A detector's per-hand scale cannot recover finger or wrist depth. Keeping
    # this offset separate from source landmarks avoids driving palms into skin.
    output[..., 2] = .235 + np.tanh(xyz[..., 2] * 1.4) * .035
    return output


def _fallback_rest_hand(side: int) -> np.ndarray:
    sign = 1.0 if side == 0 else -1.0
    wrist = np.asarray((sign * .39, .69, -.015))
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = wrist
    # Metacarpal directions measured from the existing MakeHuman right-hand bind
    # skeleton, expressed along its middle metacarpal. Avoid an artificial wide fan.
    bases = ((.51871, .78949, .32809), (.24650, .96912, -.00613),
             (0, 1, 0), (-.20546, .97843, -.02166), (-.41436, .91009, -.00687))
    yaw = np.deg2rad(60)
    basis = np.array([[-sign*np.cos(yaw), 0, -sign*np.sin(yaw)],
                      [0, -1, 0], [np.sin(yaw), 0, -np.cos(yaw)]])
    # ponytail: relaxed flexion is an animation preset; measured signer rest modes win when available.
    bends = ((12, 37, 49), (16, 46, 60), (20, 55, 70), (24, 64, 80))
    thumb = ((.2, .85, .48), (.08, .82, .56), (-.1, .8, .59))
    for finger, base in enumerate(bases):
        root = 1 if finger == 0 else 5 + (finger - 1) * 4
        parent = 0
        for joint in range(4):
            child = root + joint
            length_index = HAND_EDGES.index((parent, child))
            if joint == 0:
                direction = np.asarray(base)
            elif finger == 0:
                direction = np.asarray(thumb[joint-1])
            else:
                angle = np.deg2rad(bends[finger-1][joint-1])
                direction = np.array([0, np.cos(angle), np.sin(angle)])
            direction = basis @ (direction / np.linalg.norm(direction))
            hand[child] = hand[parent] + direction * HAND_BONE_LENGTHS[length_index]
            parent = child
    return hand


def estimate_rest_reference(
    archives: list[tuple[str, np.ndarray]], *, modes: int = 3,
) -> RestReference:
    """Estimate diverse idle poses from one signer's train-only observations.

    A candidate requires both hands to be detected, the candidate hand to sit below
    the signing space and move less than the other hand. No synthetic sway is added.
    """
    candidates: list[list[np.ndarray]] = [[], []]
    used = set()
    for path, raw in archives:
        value = np.asarray(raw, dtype=np.float32)
        if value.ndim != 3 or value.shape[1:] != (61, 5):
            continue
        complete = [(value[:, start:start + 21, 3] > 0).all(axis=1) for start in (0, 21)]
        speed = [np.r_[np.inf, np.linalg.norm(np.diff(value[:, start, :2], axis=0), axis=1)] for start in (0, 21)]
        for side, start in enumerate((0, 21)):
            take = (complete[side] & complete[1 - side]
                    & (value[:, start, 1] > .25)
                    & (speed[side] < .04)
                    & (speed[1 - side] > speed[side] + .01))
            if take.any():
                candidates[side].extend(value[take, start:start + 21, :3])
                used.add(path)

    selected = []
    for side, rows in enumerate(candidates):
        if not rows:
            selected.append(np.empty((0, 21, 3), dtype=np.float32))
            continue
        values = np.asarray(rows, dtype=np.float32)
        wrists = values[:, 0, :2]
        chosen = [int(np.argmin(np.linalg.norm(wrists - np.median(wrists, axis=0), axis=1)))]
        while len(chosen) < min(modes, len(values)):
            distance = np.linalg.norm(wrists[:, None] - wrists[chosen][None], axis=2).min(axis=1)
            distance[chosen] = -1
            chosen.append(int(np.argmax(distance)))
        selected.append(values[chosen])
    return RestReference(
        (selected[0], selected[1]),
        (len(candidates[0]), len(candidates[1])),
        tuple(sorted(used)),
    )


def _rebuild_hand(source: np.ndarray, wrist: np.ndarray, side: int, world_xyz=None) -> np.ndarray:
    # MediaPipe's source-video world estimate has per-joint depth; Apple log-scale
    # does not. Conversion changes axes only, and never feeds the classifier.
    mapped = _source_to_world(source) if world_xyz is None else np.asarray(world_xyz) * (1., -1., -1.)
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[0] = wrist
    fallback = np.asarray((.35 if side == 0 else -.35, .75, 0), dtype=np.float32)
    for edge_index, (parent, child) in enumerate(HAND_EDGES):
        direction = mapped[child] - mapped[parent]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-5:
            direction = fallback
            norm = float(np.linalg.norm(direction))
        hand[child] = hand[parent] + direction / norm * HAND_BONE_LENGTHS[edge_index]
    return hand


def _solve_elbow(shoulder: np.ndarray, wrist: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    delta = wrist - shoulder
    distance = float(np.linalg.norm(delta))
    maximum = UPPER_ARM_LENGTH + FOREARM_LENGTH - 1e-5
    minimum = abs(UPPER_ARM_LENGTH - FOREARM_LENGTH) + 1e-5
    if distance < 1e-6:
        delta = np.asarray((.01 if side == 0 else -.01, -.2, 0), dtype=np.float32)
        distance = float(np.linalg.norm(delta))
    clamped = np.clip(distance, minimum, maximum)
    wrist = shoulder + delta / distance * clamped
    axis = (wrist - shoulder) / clamped
    along = (UPPER_ARM_LENGTH ** 2 - FOREARM_LENGTH ** 2 + clamped ** 2) / (2 * clamped)
    height = np.sqrt(max(0.0, UPPER_ARM_LENGTH ** 2 - along ** 2))
    # A fixed downward/outward pole removes the old two-solution flip and the
    # raised chicken-wing elbow when a wrist passes below the shoulder.
    outward = 1 if side == 0 else -1
    pole = np.asarray((outward * .18, -.35, -.08), dtype=np.float32)
    perpendicular = pole - np.dot(pole, axis) * axis
    if np.linalg.norm(perpendicular) < 1e-5:
        perpendicular = np.asarray((1, 0, 0), dtype=np.float32)
    perpendicular /= np.linalg.norm(perpendicular)
    elbow = shoulder + axis * along + perpendicular * height
    return elbow.astype(np.float32), wrist.astype(np.float32)


def retarget_avatar(
    animation_xyz: np.ndarray,
    animation_presence: np.ndarray,
    observation_presence: np.ndarray,
    metadata: dict | str,
    *,
    rest_reference: RestReference | None = None,
    hand_world_xyz: np.ndarray | None = None,
    handshape_annotations: dict | None = None,
) -> RetargetedAvatar:
    """Retarget sparse signing motion onto a fixed-length human rig."""
    xyz = np.asarray(animation_xyz, dtype=np.float32)
    presence = np.asarray(animation_presence, dtype=bool)
    observations = np.asarray(observation_presence, dtype=bool)
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if xyz.ndim != 3 or xyz.shape[1:] != (61, 3):
        raise ValueError("animation_xyz must be [frames,61,3]")
    if presence.shape != xyz.shape[:2] or observations.shape != presence.shape:
        raise ValueError("presence arrays must align with animation_xyz")
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("avatar source must be finite and non-empty")
    if hand_world_xyz is not None and np.asarray(hand_world_xyz).shape != (len(xyz), 2, 21, 3):
        raise ValueError("world hand estimates must be [frames,2,21,3]; NaN denotes unavailable hands")

    states, observed = hand_states_from_metadata(len(xyz), metadata, observations)
    shoulders = np.repeat(np.asarray([[[.1677125, 1.285, 0], [-.1677125, 1.285, 0]]], dtype=np.float32), len(xyz), axis=0)
    elbows = np.empty_like(shoulders)
    hands = np.empty((len(xyz), 2, 21, 3), dtype=np.float32)

    targets = np.zeros((len(xyz), 2, 3), dtype=np.float32)
    for side, start in enumerate((0, 21)):
        active = states[:, side] != "rest"
        source_wrist = _source_to_world(xyz[:, start])
        # Missing detector coordinates are zero sentinels, not shoulder-height targets.
        known = np.flatnonzero(presence[:, start])
        if len(known):
            missing = np.flatnonzero(~presence[:, start])
            for axis in range(3):
                source_wrist[missing, axis] = np.interp(missing, known, source_wrist[known, axis])
        else:
            source_wrist[:] = _fallback_rest_hand(side)[0]
        source_wrist = _smooth(source_wrist)
        modes = None if rest_reference is None else rest_reference.poses[side]
        if modes is not None and len(modes):
            world_modes = np.stack([_rebuild_hand(value, _source_to_world(value[:1])[0], side) for value in modes])
            rest = np.empty((len(xyz), 3), dtype=np.float32)
            segments = np.split(np.flatnonzero(~active), np.flatnonzero(np.diff(np.flatnonzero(~active)) > 1) + 1)
            for segment in segments:
                if not len(segment):
                    continue
                neighbors = np.flatnonzero(active)
                nearest = int(neighbors[np.argmin(np.abs(neighbors - segment.mean()))]) if len(neighbors) else 0
                mode = int(np.argmin(np.linalg.norm(world_modes[:, 0] - source_wrist[nearest], axis=1)))
                rest[segment] = world_modes[mode, 0]
            states[~active, side] = "rest-grounded"
        else:
            rest = np.repeat(_fallback_rest_hand(side)[None, 0], len(xyz), axis=0)
            states[~active, side] = "rest-uncertain"
        targets[:, side] = np.where(active[:, None], source_wrist, rest)
        # Ease across state changes to remove snapping into and out of the rest pose.
        for boundary in np.flatnonzero(active[1:] != active[:-1]) + 1:
            lo, hi = max(0, boundary - 3), min(len(xyz), boundary + 4)
            alpha = np.linspace(0, 1, hi - lo, dtype=np.float32)[:, None]
            targets[lo:hi, side] = (
                targets[lo, side] * (1 - alpha) + targets[hi - 1, side] * alpha
            )

    if hand_world_xyz is not None:
        for segment in metadata.get("timeline", []):
            if segment.get("kind") != "transition":
                continue
            start, stop = int(segment["start"]), int(segment["stop"])
            if start < 1 or stop >= len(targets):
                continue
            for frame in range(start, stop):
                alpha = (frame - start + 1) / (stop - start + 1)
                ease = alpha ** 3 * (10 - 15 * alpha + 6 * alpha ** 2)
                targets[frame] = targets[start - 1] * (1 - ease) + targets[stop] * ease

    for frame in range(len(xyz)):
        for side, start in enumerate((0, 21)):
            elbow, wrist = _solve_elbow(shoulders[frame, side], targets[frame, side], side)
            elbows[frame, side] = elbow
            if states[frame, side].startswith("rest"):
                modes = None if rest_reference is None else rest_reference.poses[side]
                if modes is not None and len(modes):
                    nearest = int(np.argmin(np.linalg.norm(_source_to_world(modes[:, 0]) - wrist, axis=1)))
                    hand = _rebuild_hand(modes[nearest], wrist, side)
                else:
                    hand = _fallback_rest_hand(side)
                    hand += wrist - hand[0]
            else:
                world = None if hand_world_xyz is None else hand_world_xyz[frame, side]
                if world is not None and not np.isfinite(world).all():
                    world = None
                hand = _rebuild_hand(xyz[frame, start:start + 21], wrist, side, world)
            hands[frame, side] = hand
    if handshape_annotations:
        for segment in metadata.get("timeline", []):
            if segment.get("kind") != "gloss" or segment.get("gloss") not in handshape_annotations:
                continue
            annotation = handshape_annotations[segment["gloss"]]
            start, stop = int(segment["start"]), int(segment["stop"])
            for side in range(2):
                if not segment.get("hand_participation", [False, False])[side]:
                    continue
                hands[start:stop, side] = constrain_annotated_handshape(
                    hands[start:stop, side], annotation["selected_fingers"], annotation["flexion"])
    if hand_world_xyz is not None:
        # The world-shape audit bridges the actual endpoint hand poses, including
        # a neutral inactive hand. Wrist trajectories and arm IK remain unchanged.
        for segment in metadata.get("timeline", []):
            if segment.get("kind") != "transition":
                continue
            start, stop = int(segment["start"]), int(segment["stop"])
            if start < 1 or stop >= len(hands):
                continue
            for side in range(2):
                first, last = hands[start - 1, side].copy(), hands[stop, side].copy()
                for frame in range(start, stop):
                    alpha = (frame - start + 1) / (stop - start + 1)
                    pose = interpolate_world_hand(first, last, alpha)
                    hands[frame, side] = pose - pose[0] + hands[frame, side, 0]
    return RetargetedAvatar(shoulders, elbows, hands, states, observed)


def bone_length_metrics(rig: RetargetedAvatar) -> dict[str, float]:
    upper = np.linalg.norm(rig.elbows - rig.shoulders, axis=-1)
    lower = np.linalg.norm(rig.hands[:, :, 0] - rig.elbows, axis=-1)
    fingers = np.stack([
        np.linalg.norm(rig.hands[:, :, child] - rig.hands[:, :, parent], axis=-1)
        for parent, child in HAND_EDGES
    ], axis=-1)
    expected = np.broadcast_to(HAND_BONE_LENGTHS, fingers.shape)
    return {
        "max_upper_arm_relative_error": float(np.max(np.abs(upper - UPPER_ARM_LENGTH) / UPPER_ARM_LENGTH)),
        "max_forearm_relative_error": float(np.max(np.abs(lower - FOREARM_LENGTH) / FOREARM_LENGTH)),
        "max_hand_bone_relative_error": float(np.max(np.abs(fingers - expected) / expected)),
    }
