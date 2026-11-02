import unittest

import torch

from saseg.data.labels import LabelMapping


class LabelMappingTests(unittest.TestCase):
    def setUp(self):
        self.mapping = LabelMapping(known=(1, 2), seen_unknown=(3,), unseen_unknown=(4,))

    def test_protocol_specific_unknown_encoding(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        self.assertEqual(self.mapping.encode(labels, "A1", "eval").tolist(), [0, 1, 2, 3, -100])
        self.assertEqual(self.mapping.encode(labels, "A2", "eval").tolist(), [0, 1, 2, -100, 3])
        self.assertEqual(self.mapping.encode(labels, "A2", "train").tolist(), [0, 1, 2, 3, -100])

    def test_unmapped_labels_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unmapped"):
            self.mapping.encode(torch.tensor([9]), "A1")

    def test_overlapping_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "disjoint"):
            LabelMapping(known=(1, 2), seen_unknown=(2,), unseen_unknown=(4,))


if __name__ == "__main__":
    unittest.main()
