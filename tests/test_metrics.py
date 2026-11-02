import unittest

import numpy as np

from saseg.metrics import (
    binary_metrics,
    candidate_metrics,
    component_metrics,
    component_metrics_per_slice,
    proposal_upper_bound,
    select_f1_threshold,
)


class MetricTests(unittest.TestCase):
    def test_threshold_free_and_thresholded_metrics(self):
        target = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.2, 0.8, 0.9])
        result = binary_metrics(target, score, 0.5)
        self.assertEqual(result["auroc"], 1.0)
        self.assertEqual(result["aupr"], 1.0)
        self.assertEqual(result["f1"], 1.0)

    def test_threshold_selection(self):
        threshold, f1 = select_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertGreaterEqual(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)
        self.assertEqual(f1, 1.0)

    def test_threshold_selection_single_class_is_explicit(self):
        threshold, f1 = select_f1_threshold(np.zeros(4), np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(threshold, 0.25)
        self.assertEqual(f1, 0.0)

    def test_component_metrics(self):
        target = np.zeros((16, 16), dtype=bool)
        pred = np.zeros_like(target)
        target[2:5, 2:5] = True
        pred[2:5, 2:5] = True
        result = component_metrics(target, pred)
        self.assertEqual(result["component_f1"], 1.0)

    def test_component_metrics_per_slice_does_not_merge_slices(self):
        target = np.zeros((2, 16, 16), dtype=bool)
        pred = np.zeros((2, 16, 16), dtype=bool)
        target[0, 2:5, 2:5] = True
        target[1, 10:13, 10:13] = True
        pred[0, 2:5, 2:5] = True
        prediction = pred.copy()
        result = component_metrics_per_slice(target, prediction)
        self.assertEqual(result["component_f1"], 0.5)

    def test_candidate_metrics_purity_and_hit(self):
        target = np.zeros((16, 16), dtype=bool)
        target[2:6, 2:6] = True
        candidate = np.zeros_like(target)
        candidate[2:6, 2:6] = True
        candidate[10:12, 10:12] = True
        result = candidate_metrics(target, candidate)
        self.assertAlmostEqual(result["purity"], 16 / 20)
        self.assertEqual(result["hit"], 1.0)
        self.assertEqual(result["component_precision"], 0.5)

    def test_proposal_upper_bound_recovers_gt(self):
        score = np.zeros((16, 16))
        target = np.zeros((16, 16), dtype=bool)
        target[2:5, 2:5] = True
        score[2:5, 2:5] = 10.0
        result = proposal_upper_bound(score, target, np.ones_like(target), top_fraction=9 / 256)
        self.assertAlmostEqual(result["union_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
