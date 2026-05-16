import unittest

from CADETProcess.transform import (
    AutoTransformer,
    NormLinearTransformer,
    NormLogTransformer,
    NullTransformer,
)


class Test_Transformer(unittest.TestCase):
    def test_input_range(self):
        transform = NormLinearTransformer(0, 100)

        with self.assertRaises(ValueError):
            in_ = -10
            out = transform.transform(in_)

        with self.assertRaises(ValueError):
            in_ = 1000
            out = transform.transform(in_)

    def test_output_range(self):
        transform = NormLinearTransformer(0, 100)

        with self.assertRaises(ValueError):
            in_ = -1
            out = transform.untransform(in_)

        with self.assertRaises(ValueError):
            in_ = 2
            out = transform.untransform(in_)

    def test_no_transform(self):
        transform = NullTransformer(0, 100)
        self.assertAlmostEqual(transform.lb, 0)
        self.assertAlmostEqual(transform.ub, 100)

        in_ = 0
        out_expected = 0
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 0
        out_expected = 0
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

    def test_linear(self):
        transform = NormLinearTransformer(0, 100)
        self.assertAlmostEqual(transform.lb, 0)
        self.assertAlmostEqual(transform.ub, 1)

        in_ = 0
        out_expected = 0
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 0
        out_expected = 0
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 10
        out_expected = 0.1
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 0.1
        out_expected = 10
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 100
        out_expected = 1
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 1
        out_expected = 100
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

    def test_log(self):
        """Missing: Special case for lb_input <= 0"""
        transform = NormLogTransformer(1, 1000)
        self.assertAlmostEqual(transform.lb, 0)
        self.assertAlmostEqual(transform.ub, 1)

        in_ = 1
        out_expected = 0
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 0
        out_expected = 1
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 10
        out_expected = 1 / 3
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 1 / 3
        out_expected = 10
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 100
        out_expected = 2 / 3
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 2 / 3
        out_expected = 100
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 1000
        out_expected = 1
        out = transform.transform(in_)
        self.assertAlmostEqual(out_expected, out)

        in_ = 1
        out_expected = 1000
        out = transform.untransform(in_)
        self.assertAlmostEqual(out_expected, out)

    def test_auto(self):
        threshold = 1000

        transform = AutoTransformer(1, 100, threshold=threshold)
        self.assertTrue(transform.use_linear)
        self.assertFalse(transform.use_log)

        # Expect Log behaviour
        transform = AutoTransformer(1, 1001, threshold=threshold)
        self.assertFalse(transform.use_linear)
        self.assertTrue(transform.use_log)


if __name__ == "__main__":
    unittest.main()
