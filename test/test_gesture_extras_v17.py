"""Middle-finger novelty gesture: pose geometry, hold/latch and caption interjection."""
from pathlib import Path
import sys
import types
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.gesture_extras_v17 import MIDDLE_FINGER_TEXT, MiddleFingerGesture, is_middle_finger_hand

FINGERS = {'thumb': (1, 2, 3, 4), 'index': (5, 6, 7, 8), 'middle': (9, 10, 11, 12),
           'ring': (13, 14, 15, 16), 'little': (17, 18, 19, 20)}
X = {'thumb': -0.09, 'index': -0.045, 'middle': 0.0, 'ring': 0.04, 'little': 0.075}


def hand(extended=(), down=False, bent=()):
    """Synthetic image-space hand (y grows downward); palm length 0.1."""
    xy = np.zeros((21, 2))
    sign = 1 if down else -1
    wrist = np.array([0.5, 0.7])
    xy[0] = wrist
    for name, joints in FINGERS.items():
        base = wrist + np.array([X[name], sign * (0.06 if name == 'thumb' else 0.1)])
        xy[joints[0]] = base
        for k, j in enumerate(joints[1:], start=1):
            if name in extended:
                xy[j] = base + np.array([0, sign * 0.035 * k])
            elif name in bent:          # bent forward at the knuckle (e.g. FEEL/SICK)
                xy[j] = base + np.array([0.0, sign * 0.03]) + np.array([0.0, -sign * 0.012 * (k - 1)])
            else:                        # folded back toward the wrist
                xy[j] = base + (wrist - base) * (0.2 * k)
    return types.SimpleNamespace(xy=xy, confidence=np.ones(21))


class PoseTest(unittest.TestCase):
    def test_middle_finger_only_is_detected(self):
        self.assertTrue(is_middle_finger_hand(hand(extended=('middle',))))
        self.assertTrue(is_middle_finger_hand(hand(extended=('middle', 'thumb'))))

    def test_other_hand_shapes_are_not(self):
        self.assertFalse(is_middle_finger_hand(None))
        self.assertFalse(is_middle_finger_hand(hand()))                                    # fist
        self.assertFalse(is_middle_finger_hand(hand(extended=tuple(FINGERS))))             # open palm
        self.assertFalse(is_middle_finger_hand(hand(extended=('index',))))                 # pointing
        self.assertFalse(is_middle_finger_hand(hand(extended=('index', 'middle'))))        # V / two
        self.assertFalse(is_middle_finger_hand(hand(extended=('middle',), down=True)))     # hand pointing down
        self.assertFalse(is_middle_finger_hand(hand(extended=('index', 'ring', 'little', 'thumb'),
                                                    bent=('middle',))))                     # FEEL/SICK handshape

    def test_missing_landmarks_reject(self):
        h = hand(extended=('middle',))
        h.confidence[12] = 0
        self.assertFalse(is_middle_finger_hand(h))


class HoldTest(unittest.TestCase):
    def test_fires_once_after_hold_and_rearms_after_release(self):
        g = MiddleFingerGesture()
        pose = {'left': None, 'right': hand(extended=('middle',))}
        idle = {'left': None, 'right': None}
        fired = [g.update(pose, t / 20) for t in range(30)]          # 1.5 s held
        self.assertEqual(sum(fired), 1)
        self.assertTrue(g.active)
        self.assertFalse(g.update(idle, 1.6))
        self.assertFalse(g.active)
        self.assertEqual(sum(g.update(pose, 2 + t / 20) for t in range(20)), 1)

    def test_a_brief_flash_does_not_fire(self):
        g = MiddleFingerGesture()
        pose = {'left': hand(extended=('middle',)), 'right': None}
        self.assertFalse(any(g.update(pose, t / 20) for t in range(8)))   # 0.4 s < 0.6 s hold


class CaptionInterjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.live_captions_v17 import DEFAULT_MODEL_DIR, CaptionModels
        if not (DEFAULT_MODEL_DIR / 'manifest.json').is_file():
            raise unittest.SkipTest('caption models not exported')
        cls.models = CaptionModels()

    def test_interjection_settles_the_caption_and_is_spoken(self):
        from scripts.live_captions_v17 import LiveCaptions
        c = LiveCaptions(self.models)
        for i, g in enumerate(['HELLO', 'MY', 'FRIEND']):
            c.push(g, .9, float(i))
        u = c.interject(MIDDLE_FINGER_TEXT)
        self.assertTrue(u.text.startswith('Hello, my friend'))
        self.assertTrue(u.text.endswith(MIDDLE_FINGER_TEXT))
        self.assertIn(MIDDLE_FINGER_TEXT, u.speak)
        self.assertIn('friend', u.speak.lower())
        u = c.push('HOW', .9, 5.)
        self.assertTrue(u.text.startswith('Hello, my friend'))
        self.assertIn(MIDDLE_FINGER_TEXT + ' How', u.text)
        self.assertTrue(c.finish().text.endswith('?') or c.text.endswith('.'))
        c.reset()
        self.assertEqual(c.text, '')

    def test_interjection_alone(self):
        from scripts.live_captions_v17 import LiveCaptions
        c = LiveCaptions(self.models)
        u = c.interject(MIDDLE_FINGER_TEXT)
        self.assertEqual((u.text, u.speak), (MIDDLE_FINGER_TEXT, MIDDLE_FINGER_TEXT))
        self.assertEqual(c.finish().text, MIDDLE_FINGER_TEXT)


class ParserDefaultsTest(unittest.TestCase):
    def test_app_defaults(self):
        from scripts.app_shell_v17 import parser
        a = parser().parse_args([])
        self.assertEqual(a.caption_end_seconds, 1.5)
        self.assertFalse(a.no_middle_finger)
        self.assertTrue(parser().parse_args(['--no-middle-finger']).no_middle_finger)


if __name__ == '__main__':
    unittest.main()
