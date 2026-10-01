"""Independent binary64 camera oracle and pre-heavy raw contract tests."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from formal_camera import build_camera, from_metadata
from formal_config import Resolution


def resolution(w, h, divisor=1):
    return Resolution('divisor', divisor, w, h, w // divisor, h // divisor)


class PureCameraTests(unittest.TestCase):
    def close64(self, actual, expected):
        self.assertLessEqual(abs(actual - expected), 32 * sys.float_info.epsilon * max(1, abs(expected)))

    def test_current_camera_at_three_resolutions(self):
        for d in (1, 2, 5):
            with self.subTest(divisor=d):
                c = build_camera(resolution(1280, 720, d), ('train', 0, 'images/a.png'),
                                 intrinsics=(1777.7777777777778, 1777.7777777777778, 640, 360))
                self.assertEqual((c.width, c.height, c.cx, c.cy), (1280 // d, 720 // d, 640 / d, 360 / d))
                for actual, expected in ((c.tanx, .36), (c.tany, .2025),
                                         (c.fx, 1777.7777777777778 / d),
                                         (c.fovx, 2 * math.atan(.36)), (c.fovy, 2 * math.atan(.2025))):
                    self.close64(actual, expected)
                self.assertTrue(c.raw.reader_sentinel)
                self.assertEqual((c.znear, c.zfar, c.raster_near, c.raster_far), (.01, 100., .2, None))
                with self.assertRaises(AttributeError): c.fx = 1

    def test_fov_pair_and_existing_root_adapter(self):
        for d in (1, 2):
            r = resolution(80, 40, d)
            pair = build_camera(r, ('test', 1, 'images/b.png'), fovs=(math.pi / 2, 2 * math.atan(.5)))
            adapted = from_metadata(dict(w=80, h=40, camera_angle_x=math.pi / 2), {}, r, pair.frame_key)
            for c in (pair, adapted):
                for actual, expected in ((c.fx, 40 / d), (c.fy, 40 / d), (c.tanx, 1), (c.tany, .5)):
                    self.close64(actual, expected)
                self.assertFalse(c.raw.reader_sentinel)

    def test_anisotropic_pair_and_odd_center(self):
        r = resolution(81, 41)
        a = build_camera(r, ('train', 0, 'a'), intrinsics=(54, 82, 40.5, 20.5))
        b = build_camera(r, a.frame_key, fovs=(2 * math.atan(.75), 2 * math.atan(.25)))
        for name in ('fx', 'fy', 'cx', 'cy', 'tanx', 'tany'):
            self.close64(getattr(a, name), getattr(b, name))
        self.assertEqual((a.tanx, a.tany), (.75, .25))

    def test_complete_inheritance_without_fieldwise_fallback(self):
        r = resolution(20, 10)
        fields = dict(fl_x=15, fl_y=30, cx=10, cy=5)
        for root, frame, source in ((dict(w=20, h=10, **fields), {}, 'root'),
                (dict(w=20, h=10), fields, 'frame'),
                (dict(w=20, h=10, **fields), dict(w=20, h=10, **fields), 'both')):
            c = from_metadata(root, frame, r, ('train', 0, 'a'))
            self.assertEqual(c.raw.source, source)
            self.assertEqual(dict(c.raw.root_fields), root)
            self.assertEqual(dict(c.raw.frame_fields), frame)
        bad_pairs = [
            (dict(w=20, h=10, **fields), {'fl_x': 15}),
            (dict(w=20, h=10, fl_x=15), fields),
            (dict(w=20, h=10, **fields), dict(fields, fl_y=15)),
            (dict(w=20, h=10, **fields), {'w': 20}),
            (dict(w=20, **fields), {'w': 20, 'h': 10}),
            (dict(w=20, h=10, **fields), {'w': 40, 'h': 10}),
            (dict(w=20, h=10, **fields, camera_angle_x=1.), {}),
            (dict(w=20, h=10, camera_angle_x=1.), fields),
            (dict(w=20, h=10, **fields, camera_angle_x=-1.), {}),
            (dict(w=20, h=10, **fields), {'camera_angle_x': 1.}),
            (dict(w=20, h=10, **fields, camera_angle_y=1.), {}),
            (dict(w=20, h=10, **fields, fx=15), {}),
            (dict(w=20, h=10, **fields, znear=.01), {}),
            (fields, {}), (dict(w=20, h=10), {})]
        for root, frame in bad_pairs:
            with self.subTest(root=root, frame=frame), self.assertRaises(ValueError):
                from_metadata(root, frame, r, ('train', 0, 'a'))

    def test_invalid_numeric_center_dimensions_clipping_and_pairs(self):
        r = resolution(20, 10)
        for bad in (True, '15', None, 0, -1, math.inf, math.nan, 1e-300, 1e300):
            with self.subTest(focal=bad), self.assertRaises(ValueError):
                build_camera(r, (), intrinsics=(bad, 15, 10, 5))
        for cx in (math.nextafter(10., 11.), math.nextafter(10., 9.), 9, True):
            with self.subTest(cx=cx), self.assertRaises(ValueError):
                build_camera(r, (), intrinsics=(15, 15, cx, 5))
        for angle in (-1., 0., math.pi, math.inf, math.nan, True, 5e-324, 1e-300):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                build_camera(r, (), fovs=(angle, 1.))
        for bad in (r._replace(raw_width=True), r._replace(raw_height=0),
                    r._replace(raw_width=2147483648), r._replace(divisor=3),
                    r._replace(width=10), r._replace(divisor=0)):
            with self.subTest(resolution=bad), self.assertRaises(ValueError):
                build_camera(bad, (), intrinsics=(15, 15, 10, 5))
        for near, far in ((0, 100), (.01, .01), (.02, 100), (.01, 200), (True, 100), (.01, math.inf)):
            with self.subTest(near=near, far=far), self.assertRaises(ValueError):
                build_camera(r, (), intrinsics=(15, 15, 10, 5), znear=near, zfar=far)
        for kw in ({}, {'fovs': (1.,)}, {'intrinsics': (15, 15, 10)},
                   {'fovs': (1., 1.), 'intrinsics': (15, 15, 10, 5)}):
            with self.subTest(kwargs=kw), self.assertRaises(ValueError): build_camera(r, (), **kw)


if __name__ == '__main__':
    print('test executable:', sys.executable)
    unittest.main()
