import unittest
import numpy as np
from cub_paper_segmentation import select_and_merge


class RegionTests(unittest.TestCase):
    def test_merge_retains_ambiguous_identity(self):
        a = np.zeros((10,10), bool); a[2:5,2:5] = True
        result = select_and_merge([a,a], ["bird head","bird eye"], 3)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]["labels"], ["bird head","bird eye"])

    def test_filters_do_not_manufacture_absence(self):
        a = np.ones((10,10), bool)
        self.assertEqual(select_and_merge([a], ["bird"], 3), [])
        self.assertEqual(select_and_merge([~a], ["eye"], 3), [])

    def test_disjoint_parts_stay_separate(self):
        a = np.zeros((10,10), bool); a[:2,:2] = True
        self.assertEqual(len(select_and_merge([a,np.flip(a)], ["a","b"], 3)),2)


if __name__ == "__main__": unittest.main()
